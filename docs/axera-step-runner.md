# Running the ResNet18 training step with its covered nodes on the AX650

`coverage_report` counts the nodes of the training step
(`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`, 1,104 nodes, batch 16,
distillation loss plus Adam) that our emitters can produce at the step's
predicted calibration (`docs/axera-step-real-calibration.md`). This note is
about actually running the step that way, on the card, and measuring what
comes out.

## The pieces

- `scripts/axera/vm/axcl_batch_runner.c` is a persistent runner built inside
  the LXD guest. One `axclInit`, one device context, and line commands on
  stdin: `LOAD` a model, `RUN` it on input files, `UNLOAD` it. Tensors travel
  as raw files on the guest's virtiofs share (`/mnt/share`), not through the
  pipe. Loading a model takes about 13 ms, and running the Relu health model
  about 25 ms end to end. `axcl_run_model` through the harness instead costs
  a process, a runtime init and an `lxc file push` per call.
- `scripts/axera/axcl_session.py` is the host side (`AXSession`). It holds
  `/tmp/axcl-device.lock` for the session, so other device work queues behind
  it. It also provides `health_check`: the native `x128,y128` Relu template
  on [-1, 1] data, which must come back within 1 LSB.
- `scripts/axera/step_runner.py` turns the step into segments and runs them.
  - Every node `plan_at_calibration` covers becomes an NPU segment: its
    template, retargeted to the predicted calibration.
  - A covered live-operand MatMul runs its whole chain template.
  - Greater/Less -> Cast runs as its pair template.
  - A bias-flatten Reshape runs fused with its ReduceSum (#1908).
  - Same-shape float32 Add at exactly `[16, 1000]` uses its checked-in
    Pulsar2 FP32 template. The indexed FP32 corpus also covers the exact
    Mul/Div input and output shape signatures of the training graph's former
    binary fallbacks, including broadcasted initializer operands. These
    templates have no quantize/dequantize boundary; each signature is
    selected by op and exact input/output shapes and was compared with NumPy
    on AXCL. Uncaptured signatures retain their existing path.
  - The ResNet18 loss-tail one-hot `Mul_20` and broadcast `Sub_24` use
    checked-in native templates at their exact measured calibration. The
    Sub template expands `[16, 1]` to `[16, 1000]`; both routes are refused
    if their tensor scales or zero points change.
  - `Div_0`, `Div_1`, and `Div_34` divide `[16, 1000]` tensors by the
    constant `2`. Pulsar2 folds each into a native multiply program; the
    runner selects one of three checked-in templates by the exact measured
    input/output zero point and scale. Other Div constants or calibrations
    still require a general native template before they leave the host.
  - The crop-mask normalization `Div_453` uses a checked-in FP32
    `Expand -> Max(count, 1) -> Div` model. Float execution has counts in
    `[1,9]`; quantized mask inputs can produce empty positions, where both
    numerator and count are zero. The explicit clamp makes those positions
    zero and avoids both implicit broadcast in AxDiv and `0/0`. The exact
    `[1024,9,3136]` template passed an AX8850 VM test including zero counts;
    the step runner also verifies this segment against its matching safe-div
    simulation.
  - The checked-in exact S16 overrides are enabled by default; a different
    JSON file can be selected with `--precision-overrides overrides.json`.
    They select only templates whose full scale and
    zero-point tuple matches the fixture index; a mismatch raises instead of
    silently reverting to ORT. The current range-matched fixtures cover
    `Mul_5`, `Mul_11`, `Mul_25`, `Mul_33`, `Mul_46`, `Mul_68`, `Mul_91`, and
    `Mul_113`, `Mul_148`, `Mul_170`, `Mul_193`, `Mul_215`, `Mul_250`,
    `Mul_272`, `Mul_295`, `Mul_317`, `Mul_352`, `Mul_374`, `Mul_397`, and
    `Mul_419` at the committed ResNet18 calibration. Twelve shared-scale S16
    templates cover 21 matrix-shaped `lr * tensor` optimizer multiplies. Five
    more templates cover all 21 rank-1 optimizer multiplies by packing vectors as
    `[1,N]` for Pulsar and restoring the logical `[N]` output shape in the
    runner. A range-matched S16 Sub template also covers `Sub_32`, expanding
    its `[16,1]` input to `[16,1000]` on-device. With supported overrides,
    planning goes from 193 to 126 host nodes: 62 Mul and one Sub override are
    selected (Mul fallbacks fall from 146 to 84; Sub from 3 to 0), and two
    singleton Div nodes are guarded/folded. The two large `1-Cast(Greater)` /
    `1-Cast(Less)` masks now use swapped inclusive comparisons on-device; this
    preserves equality behavior for finite inputs.
    Two scalar constant-first Div nodes also fold when `batch_size` is exactly
    the calibrated singleton; runtime guards preserve the ORT path if it differs
    (`Div` fallbacks 44 to 42). All 17 shared optimizer templates and the
    `Sub_32` and both comparison-complement templates passed AXCL VM checks.
    Calibration metadata is in
    `fixtures/binary_op_precision/index.json`.
  - Everything else runs on the host, one node at a time in onnxruntime.
  - Segments exchange float32 tensors. Quantized templates quantize and
    dequantize internally; the FP32 Add template consumes and returns float32
    directly.
  - Every NPU segment is also simulated on the host on the same inputs:
    inputs fake-quantized at the predicted parameters, the float ops, and
    the output fake-quantized for quantized templates; unquantized templates
    run directly in float. The runner then compares device against simulation
    in output LSBs, and both against float.
  - With `--health-every 1`, the default, the health model runs after every
    device segment, so a bad model is caught at the segment that broke the
    card.
- `tinygrad_ax_backend.register_ax_device()` makes `"AX"` a tinygrad device.
  - `AXAllocator` gives host-staged buffers. AXCL binds device memory to one
    loaded model's IO, so buffers are staged per call.
  - `AXProgram` loads `AXCompiler` output, or any emitted `.axmodel`, into
    the session and runs it on the card.
  - `tests/test_axera_step_runner.py` runs a Relu (an `AXCompiler` request)
    and the step's fc dX MatMul chain through tinygrad `Buffer("AX", ...)`
    objects on the device.
  - tinygrad's own scheduler cannot target the device: no renderer maps
    UOps to templates.

## What running it found

All runs are on 2026-09-24, on the AX8850 through `axcl-vm`, with the step's
first batch (`step1_ref.pkl`) and the committed calibration
(`fixtures/step_calibration/resnet18_step_calibration.json.gz`).

1. **Emitters wrote MCode without updating its size, and that took the
   card down.**
   - `misc_op_record_emit.emit_model`, `reshape_record_emit.emit_step_reshape`,
     `elementwise_scale_emit._write` and `ElementwiseScaleEdit` replaced the
     MCode initializer's bytes but not its `dims`.
   - A zero-point move re-encodes the stream to a different length:
     ReduceSum_62, `[16,1,512,4608]`, went from 120,024 to 120,088 bytes.
   - The runtime reads the size from `dims`. A standalone load fails with
     `0x80300709`. In the first whole-step run it wedged the card instead:
     firmware dead, full stack reload needed.
   - All four emitters now go through `step_recalibrate.with_mcode`.
     ReduceSum_62 then matches its simulation exactly.
   - The native template of the same shape ran fine throughout, which is
     what isolated the fault to the emitted model.
2. **The step Reshape templates compute a Relu (#1891).**
   - The step Reshape templates are `Reshape -> Relu` builds, and the
     emitted model applies the Relu.
   - Reshape_42, on a signed input: 40% of the elements are off by up to
     164 LSB, and the error relative to float is 79%.
   - 100 of the 119 Reshape segments take an input whose calibration range
     is negative. The runner marks them `unsafe` and keeps them on the host.
     The other 19 run on the NPU and match.
   - So `coverage_report`'s Reshape count overstates what is correct by
     those 100 nodes.
   - Fixed since: signed Reshapes now take `Reshape -> Identity` templates,
     which are exact on the device (`docs/axera-reshape-signed-templates.md`).
3. **Gather templates kept their own calibration.**
   - `GatherIndexEdit` moves the indices but leaves the template's
     `(1/s, s, zp)`. On the device, Gather_57 differed from float at 26% of
     its elements.
   - A Gather is passive, so the runner now retargets it like a Reshape.
     `reshape_record_emit.retarget_scale` gained `zp_regs=GATHER_ZP_REGS`,
     since a Gather writes only 0x1b10/0x1a90. It also now accepts an `s`
     lane that is not exactly `f32(1 / (1/s))`, as a Gather template stores
     141.6667 next to 0.00705882.
   - All 27 Gather segments are now within 1.83 LSB of their simulation.
4. **Five segments disagree with their simulation. Validation pass:
   `--mode npu`, safe segments only, 275 segments, 366 nodes.**
   - 270 of 275 are within 2 LSB.
   - The failures:

     | Segment | What the device did |
     |---|---|
     | `Add_976` (x128,y128,z128) | up to 115 LSB off |
     | `Reshape_371` (fused chain `ReduceSum:16x1x64x3136:axes0,3:k0:reshape64` at s_x = 1.0e-5, s_y = 4.6e-3) | up to 144 LSB off |
     | `Reshape_416` (same chain, s_x = 1.4e-5, s_y = 5.9e-3) | device fault `0x8030070C`; the health check passed right after |
     | `Log_3`, `Log_10` | 17-89 LSB off: the emitter did not move Log's `s_y` lanes. #1910 fixed this; on current master both are exact (0 LSB) |

   - Rechecked on current master: Add_976 is still 115 LSB off, and
     Reshape_371 is still 111 LSB off. Both are emitter defects to chase:
     the Add `x128,y128,z128` retarget at these scales, and the
     `...:reshape64` fused chain at an `s_y/s_x` of about 460.
     `--validated <report>` keeps failing segments on the host in later
     runs.
   - The `...:reshape64` failures were `relayout_segment` shifting a scalar
     header word (0x1000 at byte 72) when the stream re-encoded 32 bytes
     shorter; fixed, and exact on the device
     (`docs/axera-reshape-signed-templates.md`). Add_976 is still open.
5. **The loss head cannot be uint8.** The device matches the simulation
   there; the damage is from quantization itself.
   - With Softmax on the NPU, the student's probabilities, which are about
     1e-3 for 1,000 classes, quantize at `s = 3.5e-3` to mostly 0. The
     `softmax - target` gradient is then gone.
   - With every safe segment on the NPU, the median gradient cosine to
     float is -0.62. In a run with only the loss-head/misc segments on the
     NPU, keeping just Softmax on the host brings it back to 0.998.
6. **The optimizer update cannot be uint8 either.**
   - With Adam's `m`, `v`, `sqrt(v)` and `m / (sqrt(v) + eps)` on the NPU,
     the weight update is off by a factor of about 10^6.
   - A per-tensor MinMax range cannot hold `v` of about 1e-8 next to its
     maximum.
   - `--host-optimizer` keeps every node after the gradients in float.

## The numbers

"Grad" is each weight's gradient, the tensor the Adam update consumes,
against a float run of the same batch. "Update" is `w' - w` against the float
step. Medians are over the 42 trainable tensors. Device time is the sum of
`axclrtEngineExecute` wall times inside the guest.

| Run | NPU segments / nodes | Loss (float 17.0582) | Grad cos median / min | Grad rel. err median | Update cos median |
|---|---|---|---|---|---|
| every safe segment (validation pass) | 275 / 366 | 15.864 | -0.62 / -0.78 | 4.4 | -0.11 |
| minus failing segments and the loss head | 265 / 354 | 17.091 | 0.9935 / 0.19 | 0.128 | 0.46 |
| ... and the optimizer on the host | **180 / 269** | **17.091** | **0.9935 / 0.19** | **0.128** | **0.80** |

- In the last run, every one of the 180 device segments was within 2 LSB
  of its simulation, and the health check passed before the first and
  after each one (362 device runs in all).
- The two lowest gradient cosines are the 7x7 stem's weight and bias, at
  0.19. Their backward path is the longest chain of quantized ops; every
  other gradient has a cosine of 0.986 or more.
- The update cosine stays below the gradient cosine because Adam's first
  step is close to `lr * sign(g)`: a small gradient error near zero flips
  the sign.

Time for one step, the last configuration:

| | Seconds |
|---|---|
| NPU execute, all 180 segments | 1.13 |
| Wall, including host ops, per-segment simulation and float checks, file IO and 182 health checks | 63 |
| Host-only float run of the same step | 5.1 |

This is a correctness harness, not a speedup. The NPU nodes are 24% of the
graph, and the Convs, most binary ops and all of the optimizer are still on
the host.

## Reproduce

```sh
PY=/mnt/data/cache/claude-work/tg-venv/bin/python   # numpy, onnx, onnxruntime, tinygrad fork
cd scripts/axera
systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  $PY step_runner.py --mode npu --out /path/validate.json
systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  $PY step_runner.py --mode npu --validated /path/validate.json \
  --exclude '^(Softmax|Log|Neg)_' --host-optimizer --out /path/final.json
```

- `--mode float` checks the runner itself: the loss matches the reference
  to 1e-7 relative, and every gradient has cosine 1.0.
- `--mode sim` runs the whole step on the simulation. It produces NaN,
  because the fake-quantized `Div`/`Log` see zeros that the device
  saturates.
- The peak host memory is 10 GB, from the node-by-node evaluation and the
  per-segment float checks.

## FP32 binary device profile

`profile_fp32_binary.py` profiles the checked-in unquantized Add model through
the persistent AXCL guest runner. The guest reports `axclrtEngineExecute`
time; host round-trip time also includes input staging, file exchange, and the
runner protocol.

On 2026-09-28, the AX8850 in `axcl-vm` ran the `[16,1000]` FP32 Add template
for 500 measured executions after 20 warmups. The output matched NumPy Add
exactly. Device execution averaged **296.6 us** (median 291 us, p95 319 us);
host round-trip averaged 2.28 ms (median 2.22 ms, p95 2.55 ms). This is the
template's device latency, not a full training-step speedup measurement.

Reproduce from the repository root:

```sh
AXCL_LXD_VM=axcl-vm python scripts/axera/profile_fp32_binary.py \
  --warmup 20 --runs 500 --output fp32-add-profile.json
```

Before the shape expansion below, the plan left 126 nodes on ONNX Runtime,
all binary operations: 84 Mul and 42 Div.

### Captures for the remaining shapes

`capture_fp32_binaries.py` captured the 126 former host Mul/Div nodes as 55
unique `(op, input shapes, output shape)` templates (37 Mul, 18 Div). Each
build sets that binary op's `layer_configs.data_type` to `FP32`, uses the
compiled model's FP32 IO, and is rejected unless a real AXCL execution exactly
matches NumPy. The profile record for each capture is in
`fixtures/fp32_binary/index.json`; larger tensors use fewer repetitions to
limit input staging. Their mean device times range from 253 us to 48.1 ms for
Mul and 267 us to 12.3 ms for Div, reflecting tensor size and broadcast work.

Paired AX8850 measurements also show FP32 Mul beating the matching S16
templates for `[128,128,3,3]` (590 vs 673 us median) and `[1000,512]`
(661 vs 720 us). These signatures are already selected as `fp32_binary` by
the current plan. The 100-run measurements, including p95 and fixture paths,
are recorded in `fixtures/fp32_binary/device_comparisons.json`. This is a
kernel-time comparison, not a whole-step speedup claim; apparent wins with
different broadcast input shapes were excluded.

For the native scalar-broadcast Mul segments, shape-correct profiling found
nine signatures where FP32 was at least 10% faster than the S16 template.
The planner now chooses their FP32 templates with operands reordered to match
the captured model IO, replacing 18 S16 nodes. Measured median speedups range
from 1.14x to 2.16x over 40 paired runs; marginal results below the 10%
threshold remain on S16. Details are in
`fixtures/fp32_binary/native_mul_speed_profiles.json`.

The updated plan now assigns all 1,104 graph nodes to NPU segments with zero
ONNX Runtime fallback. The full training graph ran on AX8850: all 127 FP32
binary segments had zero errors, zero LSB difference, and zero difference
from the float op; health checks passed before and after the run. The full
graph took 2.74 s of AXCL engine time and 171 s wall time including host
simulation, checking, tensor staging, and health checks. Its loss was 17.555
versus 17.058 for the float reference, and the median gradient cosine was
0.0. That remaining training accuracy loss comes from the other quantized
segments; making the binary fallbacks FP32 does not fix it.

Capture and profile on the VM-backed device with:

```sh
AXCL_LXD_VM=axcl-vm python scripts/axera/capture_fp32_binaries.py \
  --refresh --runs 30
```

## Retargeting inaccurate covered binaries

The default FP32 capture list is built from nodes that the current planner
would leave on the host. `--nodes NAME,...` additionally captures named
binary nodes even when another native emitter claims them; these explicit
entries are selected by the planner and retain the step's calibrated input
and output quantization boundaries. This lets a validated FP32 arithmetic
kernel replace a failing quantized route without changing its surrounding
calibration contract.

On 2026-09-29, an AX8850 replay identified 27 distinct signatures among the
failing binary segments. Pulsar2 built and device-validated 22 directly. Its
calibrator rejected the five scalar-first Mul signatures because rank-0 input
shapes fail in calibration, so the capture path now represents scalar inputs
as `[1]` (broadcast-equivalent) in the compiled template and reshapes the
single runtime value accordingly. All five then built and matched FP32
exactly. The 27 targeted templates cover 30 named step nodes, including
`Add_976` and the five scalar Mul nodes; a focused run of those five Mul nodes
on real step data passed at 0 LSB with no NaN updates.

The next full replay, with the five scalar templates included, ran 914 NPU
segments (1,105 graph-node executions) with no device errors and no planner
host nodes. It measured 175 exact FP32 binary segments; the strict 2-LSB gate
still rejected 33 MatMul chains and 21 quantized elementwise segments. The
training loss was 16.729 vs 17.058 float, and median gradient cosine was
-0.577, so this is not yet a validated full-training result. Report:
`/tmp/axera-fp32-qio-full-next.json`.

Two high-impact elementwise failures (`Mul_705`, `Add_730`) were separately
captured as FP32 templates and matched the calibrated simulation at 0 LSB on
real step inputs; selecting just these two restored loss to within 2e-6 of
float, with median update relative error 2.3e-6. They are deliberately not
marked `prefer_fp32_nodes`: the measured end-to-end device path took 96/133 ms
per segment, versus 14/32 ms for the local host simulation, and the capture
round trips were 28/82 ms. Thus they are useful accuracy probes, but do not
meet the faster-than-host criterion for replacing fallback. MatMul-chain
calibration/emission and faster native arithmetic remain the priority.

## Stable softmax-gradient rewrite (`--stable-softmax-grad`)

`step_runner.py --stable-softmax-grad CALIB.json` applies
`rewrite_softmax_ratio_gradients` to the step, patches the per-node records,
and regenerates the calibration for the rewritten graph over the same 4-step
calibration set (cached in `CALIB.json`; about a minute the first time). The
rewrite replaces `p * (a/p - sum(a/p*p))` with `p * (a - p*sum(a))`, removing
the quantized `0/0` behind the non-finite `Log_10`/`Div_21` segments.

On 2026-09-29, on the AX8850 with `--host-optimizer`: 328 NPU segments, zero
device errors, health 0 LSB before and after, no runtime fallbacks. Median
gradient cosine against the float reference rose from -0.577 to **0.839** with
no NaN (simulation: 0.918); median update cosine 0.53 (was -0.0001 with 7
NaN updates). The loss is unchanged (16.729 vs 17.058, forward path). The
remaining gradient error is in 34 MatMul-chain segments that fail the 2-LSB
gate. The default run without the flag is unchanged.

```sh
AXCL_LXD_VM=axcl-vm $PY step_runner.py --mode npu --host-optimizer \
  --stable-softmax-grad /path/stable-calib.json --out /path/stable.json
```

## Adam update: `--fp32-optimizer`

In simulation the NaN updates come from the uint8 `Sqrt` and `Add(eps)`
segments in front of each optimizer `Div`: `sqrt(v)` is about 1e-5 against a
tensor max of 0.43, so the 1e-8 eps is lost and 509,119 of 512,000 elements
of the 1000x512 weight quantize to 0 (0/0 -> NaN). `--fp32-optimizer` never
gives optimizer nodes a quantized template: each runs as an FP32 binary
template when one is captured for its exact shapes, else on the host in float.
Simulation gives 0 NaN updates and the same update cosine as
`--host-optimizer` (0.64), with 263 optimizer nodes (Sqrt, Add-eps, scalar
Mul) still on the host for lack of FP32 templates. Not yet run on the device.

## 16-bit MatMul pilot

`pilot_matmul_u16.py` builds a bare live-operand `MatMul(x, w)` with Pulsar2
7.0-lite at U8, U16 and S16 (`quant.layer_configs` with
`op_types: ["MatMul"]`). Pulsar2 accepts U16 and S16 for MatMul. On the AX8850,
[1,64,128]x[1,128,64] has relative error against float of 1.76e-2 at U8 and
6.8e-5 at U16/S16 (about 260x lower), at comparable latency
(0.38 ms U8, 0.28 ms U16). The 16-bit axmodel is larger (5,919 vs 4,343 bytes)
but uses the same scale-lane roles as 8-bit, plus two fixed lane constants
(256.0 and 1.0, `matmul_record_emit.FIXED_LANES`) and an `npu_params`
multiplier lane of `256 * s_x * s_w / s_y` (the `mult256` role). With those,
`matmul_record_emit.recalibrate` moves one U16 build onto another exactly in
both directions (records and params), and the emitted model matches the
native held-out build bit for bit on the AX8850 (max diff 0.0, 6.5e-5 relative
error against float). This is a bare MatMul: the step's Gather/Reshape chains
and int8-symmetric input rules at 16 bits are still to be checked.

## 16-bit MatMul chains (`--u16-matmul`)

`step_runner.py --u16-matmul REGEX` rebuilds the `matmul_chain` segments whose
name matches with Pulsar2 `layer_configs` U16 (`u16_chain.py`), calibrated on
the reference batch's real tensors (one axmodel per segment, cached in
`--u16-cache-dir`). Segments still pass and return float32; a segment passes
when it is within `U16_MAX_REL` (5e-3) of the float chain. Two Pulsar2
details, both measured:

- Every op type of the chain is set to U16, and the bias `Add` is also named
  in a `layer_names` entry.
- A `Transpose`/`Reshape` that ends the chain makes Pulsar2 quantize the whole
  output path to 8 bits (the forward Conv chains stayed at 2.2e-2 error).
  `chain_model` cuts those ops off and the runner applies them on the host as
  the segment's `output_transform`; the forward `stage2_conv2` chain went from
  2.2e-2 to 3.5e-4.

On real step data the bare backward MatMuls are about 240x closer to float at
U16 (median 2-7e-2 -> 1-6e-4) for about 2.5x the device time (100 ms -> 251 ms
summed over 17 templates; `dX_MatMul_54` 14 -> 75 ms).

AX8850 replay with `--stable-softmax-grad --host-optimizer` and the 20 forward
Convs plus eight small backward MatMuls at 16-bit (28 segments; median
float error 7e-4; health 0 LSB, no runtime fallback): median gradient cosine
**0.974** (8-bit: 0.839; MatMul chains on the host in float: 0.994), median
update cosine 0.72, loss 16.660 vs 17.058 float. Not yet at 16 bits:
`conv0_fwd` (its Pulsar2 build exceeds the 30 minute timeout), `dense0_fwd`
(1.4e-2 from float, so it ran as float), and the large backward chains
(`TMPDIR` must point at disk: `/tmp` is tmpfs and the calibration tars of the
biggest chains overflow it).

### Where the remaining error is (16-bit MatMuls, AX8850 replay)

- `--exact-fp32-io` lets the FP32 binary segments pass float instead of
  re-quantizing to the step's 8-bit boundaries (11 segments, max error 0). It
  leaves the gradients where they were (median cosine 0.971).
- `--u16-kinds` extends `--u16-matmul` to other segment kinds. The forward
  loss error comes only from the `misc` segments (Softmax, Log, Neg,
  ReduceSum): simulation with `misc` on the host in float gives loss 17.073
  against 17.058, and no other kind moves it. At U16 on the device Softmax
  is 4e-4 from float, Log 1.4e-4 and ReduceSum exact.
- The remaining gradient error is the still-8-bit backward MatMul chains
  (34 segments, 4-11% each).

### `misc` at 16 bits closes the loss gap

AX8850 replay with `--stable-softmax-grad --host-optimizer --exact-fp32-io
--u16-kinds matmul_chain,misc` (61 segments at U16: the 20 forward Convs
except `conv0_fwd`, eight small backward MatMuls, and the Softmax, Log, Neg and
26 ReduceSum segments): loss **17.004 against 17.058** float (16.660 before),
median gradient cosine **0.977**, median update cosine 0.75; zero device
errors, health 0 LSB, no runtime fallback. Five 16-bit segments failed their
gate or build (`ReduceSum_460` exceeded the 30 minute build timeout) and ran as
float. The rest of the gradient error is the 34 backward MatMul chains still at
8 bits (12 of them beyond 2 LSB of their simulation).
