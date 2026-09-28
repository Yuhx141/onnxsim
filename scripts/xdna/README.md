# AMD XDNA backend

`xdna_backend.py` provides operator legality, ONNX graph partitioning, offline
kernel-manifest loading, tile selection from Strix Halo measurements, transfer
planning, and manifest-backed GEMM dispatch through XRT (`dispatch_matmul` /
`dispatch_matmul_batch` via `XDNAArtifactExecutor`, backed by IRON's
`NPUKernel`).

`tinygrad_bridge.py` is the tinygrad↔IRON conversion layer the launcher
leaves to the caller: tinygrad tensors (or NumPy arrays) are staged as IRON
NPU tensors, and `run_tinygrad_matmul` plans, dispatches, and reads back one
2-D MatMul. Verified on the Strix Halo NPU: a 512³ i8→i32 GEMM matches the
NumPy reference exactly (`scripts/xdna` tests + `tests/test_xdna_*.py`).

`describe_tensor` accepts NumPy arrays, tinygrad `Tensor`s, and IRON tensors
(their `.dtype` forms differ: strings, `DType` objects, and NumPy scalar-type
classes respectively).

## Phase 1

`xdna_backend.py` provides:

- a conservative core-operator allowlist;
- ONNX operator analysis;
- contiguous XDNA/fallback graph partitioning;
- offline kernel-manifest loading;
- an explicit execution boundary that fails until XRT artifacts exist.

The intended offline artifact is produced by IRON/MLIR-AIE and contains an
`xclbin` and instruction stream.  Those compiler dependencies do not need to
be installed in the deployed runtime.

## Next phases

1. Add Conv dispatch through the bridge (im2col staging + `plan_conv_gemm`
   metadata already exist; the fused Conv kernel is the missing artifact).
2. Add tensor lifetime planning, static-shape validation, and DMA staging.
3. Add fused activation kernels beyond MatMul/Conv post-op fusion.
4. Add tinygrad graph lowering and optional dynamic IRON compilation.

Unsupported nodes remain explicit `FALLBACK` partitions; they are never
silently reported as NPU execution.

## Benchmarking

Run the portable Phase 1 planner benchmark:

```bash
python3 scripts/xdna/benchmark.py --runs 200 --json xdna-planner.json
```

The report includes node coverage, planner latency, estimated memory traffic,
arithmetic intensity, and whether each workload has a matching offline kernel
artifact. It also reports naive versus optimized dispatch count. The planner
fuses legal elementwise post-ops into GEMM and convolution groups, reducing
intermediate writes and host/runtime launch boundaries. To check a manifest:

```bash
python3 scripts/xdna/benchmark.py --manifest kernels.json
```

`execution` is
explicitly `not_available` until the XRT launcher is implemented, so these
numbers must not be interpreted as XDNA throughput.  Once Phase 2 exists,
device timings will be added as a separate measurement rather than replacing
the planner numbers.

## ResNet codegen and Vitis AI comparison

Generate the static XDNA build manifest and benchmark the whole-array Conv
GEMMs selected from the graph:

```bash
python3 scripts/xdna/emit_resnet_manifest.py resnet.onnx resnet-xdna-build.json
python3 scripts/xdna/benchmark_resnet_kernels.py resnet.onnx /path/to/whole_array.py \
  --json resnet-xdna-kernels.json
python3 scripts/xdna/benchmark_vitis_resnet.py resnet.onnx --json resnet-vitis.json
python3 scripts/xdna/compare_resnet.py resnet-vitis.json resnet-xdna-kernels.json
```

To compile each whole-array-compatible Conv GEMM and emit artifact paths in a
manifest, run:

```bash
python3 scripts/xdna/compile_resnet_kernels.py resnet.onnx \
  /path/to/whole_array.py resnet-xdna-compiled.json \
  --artifact-dir ./resnet-xdna-artifacts --device npu2
```

This invokes the IRON example in compile-only mode and verifies that each
`.xclbin` and instruction stream was created. It prepares kernels for runtime
integration; it does not run the ONNX graph. The report names the remaining
graph-runtime work explicitly, including im2col packing, QDQ/requantization,
and residual/fallback dispatch.

For an experimental complete-graph run, pass `--compile-all` to also compile
the shapes whose padding is inefficient, then run the graph executor:

```bash
python3 scripts/xdna/compile_resnet_kernels.py resnet.onnx \
  /path/to/whole_array.py resnet-xdna-all.json \
  --artifact-dir ./resnet-xdna-artifacts --device npu2 --compile-all
python3 scripts/xdna/run_resnet_xdna.py resnet.onnx resnet-xdna-all.json \
  --warmup 1 --iters 5 --json resnet-xdna-fullgraph.json
```

The graph runner supports XDNA Conv execution and a hybrid mode that runs
small Conv layers as CPU integer GEMMs to avoid launch overhead. Fused
bottlenecks, compiled MaxPool, and compiled residual Add+ReLU+Quantize kernels
can execute on XDNA; remaining QDQ, activation, dense, and uncompiled pooling
nodes run on the host. Reports distinguish
`full_graph_xdna_conv_host_ops` from `full_graph_hybrid_conv_host_ops`; neither
means all graph operators execute on the NPU. The quicktest ResNet-50 graph has 53 Conv nodes and 91 planned
dispatch groups. The runner compares its output with ONNX Runtime's CPU result.
Large padded GEMM specs can take time to compile and add substantial compute
that the logical Conv does not need.
`--optimize-small-m` uses the AIE2P 16-row int8 tile for small spatial GEMMs
and the minimum column count needed for each output width.
The NPU2 runtime can retain 16 xclbin contexts, while this graph's exact-shape
manifest contains 20. The runner shares four selected output widths with wider
compiled artifacts, keeping the active set to 16. On the batch-1 quicktest
model, reusing prepacked constant weights brought the all-XDNA Conv path to
105.5 ms over 10 measured runs (two warmups). Applying the QDQ-folding pattern
seen in Vitis skips 90 dequantizations that are only consumed by Conv, lowering
the all-XDNA Conv path to 89.6 ms with exact CPU-reference output. Finally,
`--cpu-small-m 64` runs 52 small-spatial Conv nodes as CPU integer GEMMs and
keeps the 256-pixel stem Conv on XDNA. The runner also skips 33 duplicate host
Relu nodes and avoids copying/scanning already signed-int8, zero-point-zero
activations. The latest all-XDNA run averaged 79.0 ms (12.7 FPS), and the
hybrid run averaged 54.1 ms (18.5 FPS); both match the CPU output exactly.
Vitis averaged 1.60 ms on the same input, so these runs are about 49x and 34x
slower, respectively. The hybrid result is faster because it avoids most NPU
launches; CPU Conv execution remains its largest cost. Closing the gap
requires graph-level XDNA fusion and moving the
intermediate QDQ/residual operations into the device program, like Vitis does.

An experimental IRON program now fuses supported identity bottlenecks on
device: three Conv stages (1×1, 3×3, 1×1), their QDQ/requantization, residual Add, and
ReLU. The compiler can derive dimensions and requantization shifts from an ONNX
block and can also accept explicit dimensions for a new specialization:

```bash
python3 scripts/xdna/fused_bottleneck_design.py --dev npu2 \
  --model resnet.onnx --block /layer1/layer1.1 \
  --xclbin-path layer1_1-fused.xclbin --insts-path layer1_1-fused.insts.bin
python3 scripts/xdna/run_resnet_xdna.py resnet.onnx resnet-xdna-all.json \
  --cpu-small-m 64 --cpu-backend torch --cpu-threads 2 \
  --fused-block-prefix /layer1/layer1.1 \
  --fused-block-xclbin layer1_1-fused.xclbin \
  --fused-block-insts layer1_1-fused.insts.bin --warmup 2 --iters 10
```

It matched both `/layer1/layer1.1` and `/layer1/layer1.2` exactly at their
quantized boundaries, including their different requantization shifts and
residual scale ratios. Explicit 4×4 spatial specialization also compiles.
The generalized binder accepts batch-one identity and projection blocks with
1×1/3×3/1×1 main-path kernels, symmetric padding, even inner channels, uint8
activations with zero point 128, int8 weights, and power-of-two requantization
ratios. Identity blocks support height one and above. The graph
runner can compose multiple non-overlapping blocks in one run by repeating
`--fused-block PREFIX XCLBIN INSTS`. The quicktest graph ran with both layer1
identity blocks fused and retained exact ONNX Runtime output; the execution
counts dropped to 46 CPU Conv calls, one XDNA Conv call, and 194 host ops.

The NPU2 driver supports at most 16 hardware contexts. Adding the standalone
MaxPool artifact to 16 fused bottleneck artifacts exceeds that limit and
causes context eviction and reloads. The runner now accounts for compiled
pooling and Conv artifacts, then routes the lowest-work Conv specialization
or bottleneck to CPU when needed to stay within the context budget. The chosen
fallback blocks and context limit are recorded in the JSON report.

The fused Conv2 path now uses the AIE 2×2 INT8 MMUL schedule for stage-1 and
stage-2 blocks with at least 16 output channels per tile and 16 output pixels;
smaller late-stage tensors keep the scalar path to avoid tile-gather overhead.
The transformed constant weights are packed as contiguous K×N tiles. Exact
quantized results were verified for layer1.1 and layer2.1: 8.87 ms vs 12.08 ms
and 9.06 ms vs 10.41 ms, respectively.

With the 16-context budget, the full quicktest selected 14 fused bottlenecks,
XDNA MaxPool, one residual Add+ReLU kernel, and CPU fallback for
`/layer1/layer1.1` and `/layer1/layer1.2`. It measured 166.1 ms (2 warmups, 10
iterations) with exact ONNX Runtime output. The individual vectorized GEMM
route is still faster: all 53 Conv nodes on XDNA with host MaxPool measured
83.1 ms (2 warmups, 10 iterations), also with exact output. The previous valid
Vitis AI baseline is 1.609 ms on the same model. A fresh Vitis rerun failed
during provider initialization in the current environment, so 1.609 ms remains
the last valid measurement.

The NPU2 data mover limits a single weight descriptor to 65,532 bytes. The
fused path streams weights in bounded chunks, and the binder now covers
projection/downsample residuals with power-of-two QDQ scales. The four
projection blocks bind with tile-memory-aware skip chunks. The
`/layer2/layer2.1` identity block remains an on-device exactness checkpoint.
Small spatial identity blocks at H=2 and H=1 also matched the reference on
device. Full graph execution is still incomplete: stem convolution and the
classifier Gemm layers run on the host, and pooling padding is prepared on the
host before upload. The 91-dispatch graph schedule remains planning metadata
rather than one executable XDNA program.

An optional `--cpu-backend torch` uses PyTorch CPU Conv2d for the small-spatial
hybrid Conv layers and skips their unused im2col staging. Converted constant
weights are cached and symmetric padding is passed directly to Conv2d. Two
intra-op threads measured 15.3 ms and 16.4 ms in repeated quicktest runs (2
warmups/10 iterations), with exact output agreement against ONNX Runtime CPU.
These runs were 9.6x–10.3x slower than Vitis. It is opt-in because its float32
accumulations can round very long integer dot products on other inputs; the
default NumPy integer path retains integer accumulation semantics.

```bash
python3 scripts/xdna/run_resnet_xdna.py resnet.onnx resnet-xdna-all.json \
  --cpu-small-m 64 --warmup 2 --iters 10 --json resnet-xdna-hybrid.json

python3 scripts/xdna/run_resnet_xdna.py resnet.onnx resnet-xdna-all.json \
  --cpu-small-m 64 --cpu-backend torch --cpu-threads 2 --warmup 2 --iters 10 \
  --json resnet-xdna-hybrid-torch.json
```

The manifest records the graph dispatch schedule, Conv GEMM dimensions, and
which Conv shapes exceed the whole-array GEMM padding budget. XDNA profiling
reports per-host-op totals, Conv preparation/upload/launch/readback totals, and
the slowest individual Conv nodes. Vitis profiling is available through
ONNX Runtime and reports its provider subgraph as a fused node, plus CPU-side
input quantization and output dequantization; it does not expose timings for
individual Conv nodes inside that fused Vitis subgraph. Enable it with
`--profile-json` on `benchmark_vitis_resnet.py`. Profiling adds some overhead,
so use a separate unprofiled run for the headline latency.

The same manifest now includes `graph_programs`: maximal connected semantic
regions across QDQ edges, graph input/output boundaries, static constant
inputs, a topologically ordered instruction stream with ONNX attributes and
QDQ scale/zero-point references, internal tensor lifetimes, an estimated peak
live-buffer size, and the operator lowerings still required. This gives the
future fused runtime a graph IR plus a buffer and dependency contract. Regions are explicitly marked
`planning_only_not_executable`; current XDNA execution uses individual Conv
kernels, selected fused bottlenecks, and host-side operators.

The build manifest also emits `operation_kernels` records for the non-Conv
operators and every Q/DQ edge. Each record carries the node inputs, outputs,
attributes, available shapes, and a proposed lowering family (for example,
NCHW pooling, broadcast arithmetic, dense GEMM, or Q/DQ conversion). These are
codegen descriptors only: they identify work and parameters for a native IRON
kernel builder, but are not executable XDNA artifacts yet. Their
`descriptor_only_native_kernel_required` status must not be counted as device
coverage. The actual full-graph run still executes those operations on the
host.

Standalone INT8 ReLU now has a native AIE tile kernel and IRON design in
`kernels/relu_int8.cc` and `relu_design.py`. The compiler emits shape-specialized
XCLBIN/instruction artifacts for ReLU nodes with known output shapes and
attaches those artifact paths to their operation records. Hardware verification
passed for 1,024- and 16,384-element tensors. The graph runner does not yet bind
these artifacts; full-model ReLU execution remains covered by the existing
fused bottleneck or host path.

Residual `Add -> Relu -> QuantizeLinear` patterns with scalar uint8 Q/DQ
parameters and power-of-two input/output scale ratios now lower to one native
fixed-point kernel. It consumes the two raw quantized activation edges, applies
the two scale ratios and zero points, clamps ReLU values, rounds ties to even,
and emits the quantized output edge. The manifest compiler specializes these
artifacts by tensor size and quantization parameters. The runner dispatches
compiled kernels for residual blocks outside the fused bottleneck set, retains
their output in device memory, and includes those XCLBINs in its 16-context
budget. A 16,384-element kernel was compiled and matched the NumPy
quantization reference on the NPU. Other scale ratios and per-channel
quantization stay descriptor-only.

Scalar float32 `Mul` nodes now lower to a shape-specialized AIE kernel when one
operand is a static scalar and the tensor shape is preserved. The quicktest
ResNet's scalar is exactly `1.0`, so codegen removes it as a zero-copy view.
A non-identity `Mul` kernel was compiled and verified on the NPU against NumPy
for 2,048 elements. Broadcasts by non-scalar tensors remain unsupported.

Batch-one NCHW float32 `GlobalAveragePool` now lowers to a channel-tiled AIE
reduction. Each tile reads contiguous channel planes and emits their means;
the 2,048-channel, 7×7 shape compiled and matched NumPy on the NPU. A 1×1
spatial input is emitted as a zero-copy view.

Batch-one NCHW float32 2D `MaxPool` with unit dilation and floor output sizing
lowers to a row-streamed kernel. The graph runner applies ONNX padding with
`-inf`, uploads the padded NCHW tensor into a reusable XRT allocation, and
dispatches the compiled artifact into a reusable device output allocation. The
result stays device-resident until a host-only consumer or graph output asks
for it; that boundary uses the runner's cached readback. The kernel streams row
slabs so the full activation does not need to fit in core memory. Both the
ResNet quicktest 16×16→8×8 pool and the standard 112×112→56×56 stem pool
compiled and matched NumPy on the NPU. General `AveragePool`, MaxPool indices,
and other data types remain unsupported. Padding is currently prepared on the
host before upload, and adjacent operator fusion is future work.

`Flatten` and `Reshape` are emitted as zero-copy contiguous tensor views, with
their input/output shapes carried in the operation record. They require no AIE
instruction artifact; the future runner can preserve the same device buffer
and update only the logical shape. Static shape inference now propagates
Reshape targets, Transpose permutations, and Concat dimensions through the
graph, which also improves buffer-lifetime estimates. Transpose still requires
a data-movement kernel.

### From planned regions to device-resident execution

The connected regions above describe dependencies; they do not imply that one
XDNA kernel can execute every listed operator. A graph runner must separately
bind each planned instruction to a supported device kernel and retain each
internal tensor in an XRT allocation until its final consumer. Adjacent fused
blocks can now pass their raw quantized activation buffers directly on device
when shape and QDQ scale match. Compiled residual Add+ReLU kernels also retain
their output on device. Conv blocks outside fused kernels, the stem, classifier
Gemm, and unsupported graph operators still use host execution.

A practical implementation sequence is:

1. Fuse stem Conv, activation, quantization, and MaxPool to remove the host
   padding and upload boundary.
2. Lower the classifier Gemm layers and their Q/DQ/ReLU edges, then join the
   stem, bottlenecks, pooling, and classifier into one device-resident schedule.
3. Mark a region executable only when every instruction has a device binding
   and every internal edge has a device-resident buffer plan. Keep planned
   node coverage, executable node coverage, and measured device execution as
   separate report fields.

This staged schedule avoids treating a connected ONNX component as proof of
hardware fusion. It also gives each step a checkable milestone: exact
quantized edge values first, then no host activation copies between device
instructions, then full graph output agreement and end-to-end timing.

## Real-device benchmark

The host-backed smoke benchmark uses IRON plus XRT and verifies the result on
the actual NPU:

```bash
source /opt/xilinx/xrt/setup.sh
export MLIR_AIE_INSTALL_DIR=$HOME/.local/lib/python3.12/site-packages/mlir_aie
export PEANO_INSTALL_DIR=$HOME/.local/lib/python3.12/site-packages/llvm-aie
export PYTHONPATH=/opt/xilinx/xrt/python:$MLIR_AIE_INSTALL_DIR/python
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$MLIR_AIE_INSTALL_DIR/lib
PYENV_VERSION=3.12.13 python3 scripts/xdna/benchmark_device.py \
  --device npu2 --problem-size 1024 --runs 30 --json xdna-device.json
```

It reports first-call (compile/cache + launch) latency separately from warm
device latency. The operation is currently vector add; GEMM is the next real
benchmark because it is the relevant compute-bound optimization target.

For GEMM, use the MLIR-AIE whole-array design and collect a column sweep:

```bash
python3 scripts/xdna/benchmark_gemm.py \
  --example /tmp/mlir-aie-xdna/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py \
  --cols 1 2 4 8 --tile-m 64 --tile-k 32 --tile-n 32 \
  --json xdna-gemm.json
```

On the attached Strix Halo, a 512×512×512 i16→i32 sweep measured roughly
345, 688, 965, and 1,102 GFLOP/s for 1, 2, 4, and 8 columns respectively.
The 8-column result has higher variance in end-to-end time; use NPU time for
kernel optimization and end-to-end time for deployment decisions. The best
tile point found so far is `(m,k,n)=(64,32,32)` at about 1,150 GFLOP/s on eight
columns; larger tiles can fail AIE tile-memory allocation. Precision changes
the best tile: the current i8→i32 result is about 1,641 GFLOP/s at 512³ and
6,172 GFLOP/s at 1024³ with `(m,k,n)=(64,64,64)`, while BF16→BF16 was about
900 GFLOP/s at `(64,32,32)`.
For 1024³ i8 GEMM, i8 output measured about 5.78 TOPS-equivalent and i16
output about 4.92 TOPS; i32 accumulation remains the preferred target for both
performance and numerical range.
These are kernel measurements, not advertised peak-TOPS comparisons, and all
three runs verified their output against the reference.

Earlier scaling baseline for i8→i32 with eight columns and `(64,32,64)` tiles:

| GEMM shape | NPU throughput | End-to-end latency |
|---|---:|---:|
| 512³ | 1.57 TOPS-equivalent | 0.38 ms |
| 1024³ | 4.76 TOPS-equivalent | 0.88 ms |
| 2048³ | 6.74 TOPS-equivalent | 3.15 ms |

This confirms that the optimized kernel becomes substantially more efficient
as the workload grows; small GEMMs are dominated by launch and transfer cost.

At 1024³ with the same i8 tile, four columns measured 3.52 TOPS-equivalent
and 0.87 ms end-to-end, while eight columns measured 5.43 TOPS-equivalent and
0.58 ms end-to-end. Full-array placement is therefore the current default for
large GEMMs; narrower placement is mainly useful when sharing the NPU.

`benchmark_gemm.py` also reports utilization against a configurable peak
reference (`--peak-tops`, default 50). This is only a reference because AMD's
advertised peak depends on precision and operation counting; it is useful for
tracking progress, not as a direct cross-precision comparison.

Data-transfer baseline from the verified vector-add kernel reached about
0.36 GiB/s at 16K elements, 1.95 GiB/s at 256K, and 2.47 GiB/s at 1M
elements. The backend now selects `stream`, `double_buffered_stream`, or
`weight_resident` staging using tensor size and reuse count; this is intended
to hide DMA behind compute and avoid reloading reusable weights.
For the vector streaming kernel, a 128-element sub-tile currently gives the
best measured large-transfer bandwidth at about 2.46 GiB/s; 256 elements does
not improve it.
