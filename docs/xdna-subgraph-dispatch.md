# XDNA subgraph dispatch investigation

## Recommendation

Dispatch contiguous graph regions through one IRON runtime invocation, keeping
intermediate tensors and static weights on the NPU for the duration of that
invocation. Choose region boundaries from measured launch, transfer, compute,
and context costs. Do not use operator-count reduction as the objective.

For this ResNet workload, the first useful target is a compiled residual stage
(for example, all blocks in `layer1`) with its entry and exit as the only host
boundaries. Keep the existing per-op path as a fallback for unsupported or
unprofitable regions. The current standalone fused-bottleneck design should
not be selected automatically: its measured kernel-call cost outweighs the
dispatches it removes.

## Evidence from the current RPC runs

All numbers below use `test_model.onnx` at `[1, 3, 32, 32]`, seed 0, and matched
full-graph XDNA/Vitis AI runs on the RPC host.

| Schedule | XDNA latency | Vitis AI latency | Notes |
| --- | ---: | ---: | --- |
| Hybrid, `cpu_small_m=128`, `torch-int8` | 16.10 ms | 1.61 ms | Exact output; 1 XDNA Conv, 1 XDNA MaxPool, 52 CPU Conv calls |
| 14 fused bottlenecks | 137.01 ms | 1.60 ms | Exact output; fused kernel calls sum to 128.89 ms; two blocks fell back at the context limit |

The single native Conv path spends about 1.25 ms in dispatch and post-processing;
MaxPool spends about 0.92 ms in its kernel call. With `cpu_small_m=0`, 51 native
Conv calls spent about 74 ms in kernel calls. This makes one-launch-per-operator
dispatch uneconomical for the small feature maps in this graph.

The Vitis profile has one large `VitisAIExecutionProvider` kernel event at about
1.7 ms, bracketed by small input-quantize and output-dequantize events. Its
`vitis_npu_node_count=3` includes those boundary nodes; it should not be read as
three independently dispatched model partitions.

## Why the current “fusion” misses

The current Conv executor creates an `NPUKernel` call per Conv, uploads the
im2col activation and packed weights, and reads the result back to host memory.
Thus adjacent native Conv nodes do not form a device-resident execution region.
The graph runner's fused bottleneck avoids some host readbacks, but still calls
one separate, synchronous kernel per block. Its runtime sequence transfers the
activation, streams multiple parameter chunks, then drains the output. The
measured 5–18 ms per bottleneck shows that this implementation's dataflow and
launch path need work before increasing fusion coverage.

The existing context planner also has a hard budget of 16 active XRT contexts.
It accounts for unique xclbins, so compiling one artifact per shape/block can
consume the budget before a useful region schedule is formed.

Two runtime-level micro-experiments did not improve the fused block. Batching
all activation/parameter fills into one `TaskGroup` produced the same measured
kernel-call time as finishing each parameter transfer separately (about
11.76 ms for `/layer1/layer1.0`). Compiling its `ExternalFunction`s with
`inline=True` also showed no repeatable improvement (about 11.8 ms). Keep the
serialized transfer path for now; the evidence points to work/dataflow inside
the block runtime, not just task-group or C-call boundaries.

## Dispatch policy to implement

1. **Find legal regions.** Start with contiguous Conv/activation/QDQ sequences,
   residual branches and joins, and pooling. Require compatible quantization,
   layout, and static shapes. Keep unsupported operations at explicit region
   boundaries.
2. **Price each candidate.** Estimate
   `T = launch/context cost + boundary DMA + internal compute + synchronization`.
   Compare a candidate region with the sum of its operator launches and with
   the CPU fallback. Charge every host-visible intermediate and every context
   that the schedule keeps active.
3. **Keep device values resident.** Transfer only region inputs and outputs.
   Reuse immutable weights across inferences when the runtime allows it; stream
   them only when they cannot fit in the chosen on-chip storage plan.
4. **Schedule by latency and resources.** Favor the largest profitable region
   that fits the selected tile/memory budget and the XRT context budget. Use
   spatial parallelism for independent branches only when it reduces critical
   path time after routing, synchronization, and added context costs.
5. **Measure the actual schedule.** Report host-call time, input/weight/output
   transfer time, device compute time, context switches/loads, and boundary
   bytes separately. Compare the same input and graph output against Vitis AI.

The planner should produce a small Pareto set over latency, context count, and
boundary traffic instead of a single “most fused” schedule. For batch-one
latency, reject any fusion whose measured end-to-end launch cost is greater
than the calls and transfers it removes.

## Runtime partition assignment

The ResNet RPC report now includes `runtime_subgraphs`, one row per planned
semantic region. Each row carries the region's entry/exit values, internal
live-buffer estimate, lowering gaps, per-instruction executor assignment, and
contiguous executor segments. It distinguishes a single compiled dispatch,
multiple XDNA dispatches, hybrid host/device regions, and host-only regions.
`native_dispatch_count` counts distinct compiled units inside that planned
region, so a collection of Conv nodes using the same artifact still counts as
separate launches. The report also lists internal device/host crossing values
and edges. In the current quicktest ResNet it reports one connected planned
region, 53 native dispatches, 115 executor segments, and 209 internal
device/host crossings. Vitis AI's ONNX Runtime trace shows one large fused
provider node for the supported part of this same model. The comparison makes
the gap concrete: XDNA has useful kernels, but its graph still crosses the
host/device boundary hundreds of times instead of running as a partition.

The XRT RPC backend now keeps contexts, kernel handles, instruction BOs, and
argument BOs across requests in one serialized server process. It also reports
context-cache hits and misses. The standalone bottleneck artifacts still have
different XRT memory groups at adjacent boundaries, so these boundaries stay
host-staged until they are compiled into a compatible shared region.

## Next experiments

- Split the bottleneck timing into host setup, weight transfer, activation
  transfer, core execution, and output drain. The current `kernel_call_ms`
  combines these costs and cannot identify which part makes the standalone
  fused design slow.
- Prototype one `layer1` region in a **single** runtime/xclbin, linking the
  blocks with device-side FIFOs or buffers. Compare it with per-op Conv and
  per-block bottleneck dispatch using identical inputs.
- Benchmark region candidates at each residual-stage boundary. Track launch
  count, host-visible bytes, unique xclbins, context-budget fallbacks, and
  median latency; do not assume a bottleneck or stage is profitable from node
  count alone.
- Keep `torch-int8` CPU fallback as a measured option for small-M operators;
  do not count those operators as XDNA work in coverage or performance claims.

## Hardware/runtime references

- [IRON programming guide](https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/README.md): `Runtime` sequences describe host `fill`/`drain`; `ObjectFifo` and tile-local `Buffer` describe device-side movement and storage.
- [MLIR-AIE device descriptions](https://github.com/Xilinx/mlir-aie/blob/main/docs/Devices.md): NPU2 is an 8-column, 6-row array with shim DMA, memory, and compute tile rows. Region designs must account for placement and tile memory, not just graph operators.
- [MLIR-AIE configuration guide](https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/iron_configuration.md): the XRT runtime caches hardware contexts and exposes `XRT_CONTEXT_CACHE_SIZE` for that cache.
- [Ryzen AI deployment documentation](https://ryzenai.docs.amd.com/_/downloads/en/latest/pdf/): the Vitis AI execution provider partitions the ONNX graph and executes supported subgraphs on the NPU, which is the right comparison point for region-level dispatch.

## Device-side link prototype status

A three-bottleneck IRON prototype is available through RPC as `fused_stage`. It
connects each block's output FIFO to the next block's activation FIFO and keeps
only the first activation, packed parameters, and final output host-visible.
The compiled MLIR contains both links, and the parameter DMA offsets advance
through each packed block in order.

**Root cause found and fixed.** The linked stage was wrong because the per-block
worker functions in `linked_bottleneck_stage_design.py` were closures defined in
the block loop. IRON traces worker bodies after the loop ends, so every block ran
with the *last* block's `chunks1/chunks2/chunks3/skip_chunks`. Identity->identity
links happened to be exact (both blocks have `skip_chunks=0`), but a projection
block followed by anything ran with the wrong weight-stream shape and its
consumers read garbage. The worker factory `_block_workers` now binds the counts.
Bisect that located it: 1 block exact; projection->identity max error 127
(pixel-constant, bias-only-looking output); identity->identity exact.

After the fix, linked `layer1.0`->`.1` and `.0`->`.1`->`.2` are bit-exact against
the CPU reference at every compared boundary. The design now accepts one to three
blocks (`--blocks A [B [C]]`, runner `--fused-stage BLOCK... XCLBIN INSTS`), and
`--tap` (with `ONNXSIM_XDNA_STAGE_TAP=1` in the runner) exists as a debug hook
(note: broadcasting the linked FIFO to a host drain currently times out on the
device, so it is not yet a usable tap).

Measured: the linked three-block `layer1` stage takes 16.7 ms per launch, versus
24.9 ms for three separately dispatched fused blocks (11.1 + 6.9 + 6.9). It is
still far from Vitis AI's ~1.6 ms whole-model latency: the stage streams 167 KB
of weights from host DDR one awaited 18.5 KB chunk at a time, so weight
streaming, not compute, is the next target (keep weights resident in L2/memtile
across inferences, or issue chunk fills without a per-chunk await).

The Vitis capture adds selected quantized tensors as ONNX graph outputs and
runs them through a separate Vitis AI session. RPC XDNA capture saves linked
stage or standalone fused-block outputs to NPZ files when `capture_outputs` is
enabled. These captures establish that the mismatch appears only with the
linked stage, but the current stage artifact exposes only its final boundary.
The next diagnostic is to tap intermediate linked-stage FIFOs or build a
temporary host-drained stage variant so the first failing boundary can be
identified. Keep the RPC `fused_stage` path experimental until those boundary
comparisons are exact.
