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

The graph runner supports an all-XDNA Conv mode and a hybrid mode that runs
small Conv layers as CPU integer GEMMs to avoid launch overhead. In both modes,
QDQ, activation, pooling, residual, and dense nodes run on the host. Reports
distinguish `full_graph_xdna_conv_host_ops` from
`full_graph_hybrid_conv_host_ops`; neither means all graph operators execute on
the NPU. The quicktest ResNet-50 graph has 53 Conv nodes and 91 planned
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

An experimental IRON program now fuses one supported identity bottleneck on
device: three 3×3/1×1 Conv stages, their QDQ/requantization, residual Add, and
ReLU. It is currently specialized for the quicktest model's
`/layer1/layer1.1` block (NPU2, 8×8×256 input, 64 inner channels, fixed
quantization). Compile it and pass its artifacts to the graph runner:

```bash
python3 scripts/xdna/fused_bottleneck_design.py --dev npu2 \
  --xclbin-path layer1_1-fused.xclbin --insts-path layer1_1-fused.insts.bin
python3 scripts/xdna/run_resnet_xdna.py resnet.onnx resnet-xdna-all.json \
  --cpu-small-m 64 --cpu-backend torch --cpu-threads 2 \
  --fused-block-prefix /layer1/layer1.1 \
  --fused-block-xclbin layer1_1-fused.xclbin \
  --fused-block-insts layer1_1-fused.insts.bin --warmup 2 --iters 10
```

The fused block matched the ONNX Runtime quantized boundary exactly in an
isolated check. In a full quicktest graph run it also preserved exact final
output agreement; the graph reported one fused block, 49 CPU Conv calls, one
XDNA Conv call, and 201 host operators. Three warmed runs averaged 24.1 ms,
with 4.6 ms attributed to the fused block including its call and readback.
This is an executable correctness milestone, not a graph-wide speedup: the
kernel is a scalar integer implementation and only one identity block is
lowered this way. Remaining fusion work includes optimizing this kernel,
supporting downsample/stride-changing residual blocks and other shapes or
quantization layouts, and replacing per-node CPU/host execution with a generic
graph-region compiler and scheduler. The existing 91-dispatch graph schedule
is still planning metadata; the runner does not yet turn those regions into
one executable XDNA program.

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
`planning_only_not_executable`; current XDNA execution still uses individual
Conv kernels and host-side operators.

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
