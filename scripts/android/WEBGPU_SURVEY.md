# WebGPU (Vulkan) on the Snapdragon 8+ Gen 1 phone: implementation survey

Target: SM8475 (Adreno 730), `vulkan.adreno.so`, Vulkan 1.3 (conformance 1.2.2).
Question: can we run ONNX models through a WebGPU implementation on its Vulkan backend?

Provenance: a web-only survey. Only a few pages were read in full. Items marked
*unverified* come from memory or search snippets. No Adreno 730 numbers were found for any
WebGPU runtime, so performance has to be measured on the device.

## Comparison

| Stack | State (2026) | Android/Vulkan | ONNX maturity | Adreno notes |
|---|---|---|---|---|
| ORT WebGPU EP (Dawn + Tint) | Active (Microsoft, Google) | Listed as supported; `tools/ci_build/github/android/default_full_aar_build_settings.json` passes `--use_webgpu` | Best of the group: broad ops, MatMulNBits and LLM kernels | Real bugs, see below |
| Dawn alone | Active | Vulkan backend | Not an ML runtime; it is what ORT uses | Same bugs (Tint: WGSL to SPIR-V) |
| wgpu / wgpu-native (Rust, naga) | Very active | Vulkan | No ONNX runtime | Adreno 730 storage-buffer corruption reported downstream (bevy #24926); f16 and subgroups are opt-in |
| wonnx | Archived 2025-05 | Vulkan | ~100 ops, no int64, static shapes only | Dead; not viable |
| burn (CubeCL wgpu/Vulkan) | Active | Vulkan, WebGPU | Own format; ONNX only via import converter | No Adreno data |
| tract, candle | CPU-centric | No GPU path found | tract ONNX coverage good on CPU | n/a |
| emdawnwebgpu | Browser port of Dawn's webgpu.h | Not for native Android | n/a | n/a |
| ncnn / MNN (non-WebGPU reference) | Active | Mature on Adreno (Vulkan/OpenCL) | Converters | Hand-tuned; Adreno OpenCL usually faster than Vulkan (*unverified*) |
| tinygrad, TVM (non-WebGPU reference) | Active | tinygrad: QCOM/OpenCL; TVM: Vulkan/OpenCL | TVM ONNX frontend solid | Romou (MobiCom'22): 6.6-9.1x over stock TVM on Adreno |

Not researched: Diligent, sokol, webgpu-native headers (graphics/header layers, no ML runtime).

## Known Adreno / Dawn problems

- **Subgroups crash on Adreno 750.** ORT 1.30's MatMulNBits subgroup-shuffle path segfaults
  Qualcomm's shader compiler (`libllvm-qgl.so`). Adreno 660 is unaffected. Workaround: create the
  device without the `subgroups` feature. https://github.com/musetric/musetric/issues/901
- **Adreno 640 subgroups.** `subgroupBroadcast` makes `CreateComputePipelines` fail.
  https://issues.chromium.org/issues/351745820
- **Adreno 7xx pipeline failures** (Chrome 153): 8 of 23 compute pipelines fail with
  `VK_ERROR_UNKNOWN`. The shared trigger is a WGSL struct that is both the element type of a
  `var<workgroup>` array and stored whole into a `var<storage, read_write>` array. Workaround:
  split the struct into scalars. https://github.com/linebender/vello/issues/1957,
  https://github.com/francisdb/cuelight/issues/237
- **Adreno 6xx with Vulkan 1.1**: Dawn adapter creation fails (missing extensions). Our 730 has
  a 1.3 driver, so this likely does not apply.
- **Adreno 750 f16 miscompile**: f16vec4 prefetch arrays carried across loops next to fp32
  accumulators (SimdPaddleOCR PR #22). Unverified on the 730.
- llama.cpp Vulkan has Adreno compiler bugs and large-batch failures (#5186, #8743).
- Not found: f16, `packed_4x8_integer_dot_product` and max-workgroup-size limits for the 730.
  Query them on the device (`vulkaninfo`).

## Recommendation / plan

1. Use the ORT WebGPU EP. It is the only ONNX-capable WebGPU path worth trying.
2. Build with `--android --android_abi arm64-v8a --use_webgpu` (out of tree, one heavy job at a time).
3. Start with `subgroups` and `shader-f16` disabled, then enable each separately and compare
   outputs to the CPU EP to catch miscompiles.
4. Keep ncnn/MNN or the QNN path as the performance baseline; WebGPU on Adreno will probably not beat them.

## Sources

- https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
- https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/support.md
- https://github.com/webonnx/wonnx
- https://github.com/tracel-ai/burn
- https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md
- https://github.com/bevyengine/bevy/issues/24926
- https://microsoft.com/en-us/research/uploads/prod/2022/02/mobigpu_mobicom22_camera.pdf

## Measured on the phone (2026-09-29)

Build: ORT `125ea21` (main, Aug 2026), `--android --android_abi arm64-v8a --android_api 29
--use_webgpu` (static Dawn/Vulkan inside `libonnxruntime.so`, NDK r27.2, ~32 MB). Test: a
native C API program on the Adreno 730 via `adb`, WebGPU EP defaults, device features not
restricted (the EP has no option to switch off f16/subgroups).

- The EP loads and runs on Vulkan. Small graphs: ~0.9-2 ms per run for a Conv-Relu-GAP-MatMul-Softmax net.
- Correct on device: Relu, GlobalAveragePool, MatMul (3072x10), Softmax, Transpose `[0,1,3,2]`.
- **Wrong on device: `Transpose perm=[0,2,3,1]` (NCHW->NHWC) and therefore every Conv.**
  A lone NCHW->NHWC transpose of a 1x3x32x32 tensor is 50% wrong: output columns 8-15 and 24-31
  are bad in every row and channel, columns 0-7 and 16-23 are right (an 8-wide alternating
  pattern, consistent with a tiled/workgroup kernel problem). Conv variants (3x3, 1x1, padded,
  unpadded, 1/4/16 output channels) are 59-96% wrong, max abs error up to ~1.2.
- The phone CPU EP matches the host CPU exactly on the same model, so the model and input are fine.

### Core-op sweep (235 single-op models, `webgpu_ops/`)

`webgpu_ops/gen_ops.py` builds the models (fp32, input 1x3x32x32, opset 20): unary and binary
elementwise, broadcasts, comparisons, reductions, ArgMax/Min, Softmax, CumSum, Reshape/Concat/
Split/Slice/Pad/Tile/Expand/Gather/Resize, all 24 rank-4 Transposes, pooling, norms, MatMul/Gemm,
Conv variants and ConvTranspose. Each runs on the phone and is compared to host CPU ORT.

- Default settings: **195 pass, 40 fail.**
- **Root cause: ORT's shared-memory *tiled* Transpose kernel** (`use_shared` in
  `core/providers/webgpu/tensor/transpose.cc`: 16x16 workgroup, `var<workgroup>` tile,
  `workgroupBarrier()`). It runs for any transpose that coalesces to 2-D, and for the
  NCHW<->NHWC transposes ORT inserts around layout-sensitive ops.
  - Failing explicit perms: 0231 0312 2031 2301 2310 3012 3102 3120, and a plain 2-D `[96,32]`
    transpose. The other 16 rank-4 perms and both rank-3 cases pass (they take the plain kernel).
  - Everything else that failed goes through those inserted transposes: all Conv variants,
    ConvTranspose, MaxPool, AveragePool, GlobalMaxPool, BatchNormalization, SpaceToDepth.
- **Confirmed two ways.** (1) With `preferredLayout=NCHW` (provider option key `preferredLayout`,
  not the `ep.webgpuexecutionprovider.` config name) the layout-inserted failures go away, except
  `Conv` with 1 output channel. (2) With the tiled path disabled
  (an experiment; that env-var patch has since been replaced, see the fix below) **all 40
  Transpose/Conv/Pool/BN/S2D failures pass in the default NHWC layout, including that 1-channel Conv.**
- The remaining 5 mismatches (Equal, Greater, LessOrEqual, And, Cast-to-int) were a test artifact: the probe built
  its input in float32 on the phone while the host used float64-then-round, so exact-tie thresholds differed by 1 ulp
  (the phone CPU EP showed the same). With the input written to `x.bin` and read by both sides, all 235 match.
- **Host reference (same ORT revision, unpatched, x86-64 build):** the RTX 5050 (NVIDIA
  driver), the Radeon 8060S (RADV) and lavapipe (software Vulkan) each pass 230/235; the 5
  misses are the same tie artifacts. So the tiled Transpose, and everything that goes through
  it, is correct on three other Vulkan implementations, including Tint's output on them. The
  fault is specific to the Adreno 730 Vulkan stack (its driver/shader compiler) rather than ORT's
  kernel logic. Driver selected per run with `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/<x>_icd.json`.
- **Construct isolated** (`webgpu_ops/dawn_repro/`, a standalone Dawn harness plus WGSL kernels
  using the same Dawn/Tint as ORT): the tile's odd row stride. ORT declares
  `tile: array<array<f32, tile_size + 1>, tile_size>` (stride 17) and the guarded store/barrier
  pattern around it miscompiles on the Adreno 730 (Qualcomm Vulkan 512.615.0, compiler
  EV031.36.08.11). ORT's kernel is wrong even at 1x16 (50%) and 256x256 (26%); the same kernel
  with stride 16 is correct on 8 shapes; a flat stride-17 array is still wrong; the same WGSL is
  correct on the RTX 5050, RADV and lavapipe. Dropping the bounds guards also "fixes" it but is
  not general; adding `storageBarrier()` leaves 2.4% wrong (a race-like symptom).
- Tint/Dawn have Qualcomm-gated workarounds (matrix pass-by-pointer, std140 column vectors,
  `NClamp` scalarization, uniform vector component loads, command-buffer splits) but none for
  workgroup memory or barriers, so this kernel gets Tint's normal output.
- **Fix, verified through ORT:** `webgpu_ops/ort_transpose_qualcomm_unpadded_tile.patch` uses an
  unpadded tile when `adapter_info.vendor == "qualcomm"` (Dawn reports `qualcomm` / `adreno-7xx`
  here). With it the full sweep passes **230/235 on the phone at default settings**; the other 5
  are the tie artifacts. The padded tile is kept for other vendors (avoids bank conflicts).
- Not yet known: whether other Adreno generations (6xx, 8xx) share it, and the kernel-speed cost
  of the unpadded tile on Adreno.

### Ops with no WebGPU kernel (CPU fallback), now implemented

A pass in the sweep does not mean the op ran on the GPU: the WebGPU EP silently assigns unsupported nodes to
the CPU EP. `probe.cc` with `PROFILE=<prefix>` writes ORT's profile, and `placement.py` reports every node that
ran on the CPU EP. In the 235-op sweep that found eight ops without a WebGPU kernel: **Sign, Round, Softsign,
Selu, IsNaN, LogSoftmax, SpaceToDepth, LRN** (ArgMax/ArgMin/DepthToSpace are registered; the only other CPU
nodes are the test's own int64 `Cast`s, which need the EP option `enableInt64`).

`webgpu_ops/ort_webgpu_missing_ops.patch` (applies on ORT `125ea21` after the Transpose patch; new files under
`onnxruntime/core/` need `git add -f` because of a global `core` gitignore pattern) adds them:

- Sign, Round (WGSL `round` is ties-to-even like ONNX), Softsign, Selu (alpha/gamma baked into the shader and the
  cache hint), opsets as in ONNX incl. 22.
- IsNaN: float32 only, bool output; WGSL has no `isnan` and the compiler may assume no NaNs, so it tests the bit
  pattern (`bitcast<u32>(x) & 0x7fffffff > 0x7f800000`).
- LogSoftmax: the Softmax kernel with a flag: `(x - max) - log(sum)`, no clamp; opsets 1/11/13 like Softmax.
- SpaceToDepth: reuses the generic permutation program of DepthToSpace (NCHW perm `[0,3,5,1,2,4]`, NHWC
  `[0,1,3,2,4,5]`); LRN: a new per-element kernel, sum of squares over the channel window in f32, NCHW and NHWC via a
  channel stride. Both need `kMSInternalNHWCDomain` registrations too, because ORT's layout transformer rewrites them
  to NHWC on WebGPU (the first build failed at session creation with "Kernel not found: com.ms.internal.nhwc.LRN").

Verification on the phone (`gen_new_op_tests.py`, 48 more models: NaN/Inf/denormal inputs spliced in with `Where`,
exact halves and large values for Round, custom Selu alpha/gamma, block sizes 2/4/8 and rectangular/batched
SpaceToDepth, a S2D->D2S round trip, LogSoftmax on all axes / 3072-long rows / very large logits / opset 11, LRN
with sizes 1-7 at strong alpha): **283 models, 275 match host-CPU ORT; the other 8 are even-size LRN, which ORT's
CPU LRN rejects, and all of LRN (incl. those) matches a float64 numpy reference to ~2e-7.** `placement.py` then
shows 499 nodes on WebGPU and only the 6 int64 `Cast`s on CPU.

Not covered: float16 (the new kernels accept it via `WebGpuSupportedFloatTypes`, but only fp32 was run), other
Adreno generations, kernel speed (correctness only so far), and ORT's own unit tests (not built here).

