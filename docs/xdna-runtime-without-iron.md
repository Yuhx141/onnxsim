# Running XDNA artifacts without MLIR-AIE at runtime

## Finding

The compiler and runtime dependencies can be separated. Runtime execution does
not need MLIR construction or compilation when it receives a prebuilt
`xclbin` plus the matching `insts.bin`; the XRT path loads those artifacts and
submits their buffers to the NPU. The current ResNet runner nevertheless
imports IRON at runtime for both device tensors and `NPUKernel`, so it needs a
small runtime-port before MLIR-AIE can be removed from the deployed Python
environment.

The existing `scripts/xdna/README.md` already describes offline artifacts as
`xclbin` plus instruction stream and says compiler dependencies need not be
installed on the deployed runtime. The graph runner has not yet reached that
separation.

## Current runtime coupling

| Runner responsibility | Current call | Runtime replacement needed |
| --- | --- | --- |
| Device allocation and host mapping | `aie.iron.tensor`, `iron.zeros` | Small tensor wrapper over `pyxrt.bo`, mapped NumPy storage, shape/dtype metadata, sync and lifetime management |
| Kernel/context creation | `aie.utils.NPUKernel(xclbin, insts)` | Cached XRT device, xclbin, hardware context, kernel, and instruction BO |
| Kernel launch | `kernel(tensor, ...)` | Bind instruction and data BOs in the XRT kernel ABI, start/wait, check ERT status |
| Host write/read | tensor `.overwrite()` / `.numpy()` | Explicit NumPy map plus BO sync in each direction |
| Device-resident handoff | pass the same tensor to the next kernel | Preserve the underlying BO and avoid host sync between consumers |
| Device identification | `aie.iron.device.from_name("npu2")` | XRT device index/configuration; current deployment uses device index 0 |

These are concentrated in `scripts/xdna/run_resnet_xdna.py`: `_kernel`, the
workspace allocation methods, `_host_value`, and the Conv / bottleneck /
MaxPool / residual kernel call sites. The Python-only planner/backend in
`xdna_backend.py` is already independent of MLIR-AIE.

## Direct XRT pieces available on this host

The XRT installation contains `/opt/xilinx/xrt/python/pyxrt.cpython-312-...so`
and its `pyxrt.pyi` API description. The binding exposes the required device,
XCLBIN, context, kernel, buffer-object, map/read/write/sync, and run APIs. The
local MLIR-AIE `XRTHostRuntime` source shows the current artifact launch
contract:

1. Open one XRT device and register the XCLBIN.
2. Create the hardware context and the `MLIR_AIE` kernel handle.
3. Read `insts.bin` and place it in a cacheable BO using kernel argument 1's
   memory group.
4. Launch the kernel with opcode `3`, the instruction BO, instruction byte
   length, and the design's host BOs in runtime-sequence order.
5. Wait for completion and require `ERT_CMD_STATE_COMPLETED`.

The buffer group for each tensor must match the XRT kernel argument's group
requirements. Buffer ownership must keep the device, context, BO, and mapped
host view alive through completion. BO synchronization and reuse need to match
the current IRON tensor semantics; otherwise host writes can be stale on the
NPU or device results stale on the CPU.

These APIs are documented in [XRT Native APIs](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html), and XRT's official Python examples show BO synchronization and kernel execution in Python ([XRT simple Python example](https://github.com/Xilinx/XRT/blob/master/tests/python/02_simple/main.py)). MLIR-AIE's guide also treats the XCLBIN/instruction pair as loadable by its host runtime after compile artifacts have been produced ([IRON configuration and runtime](https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/iron_configuration.md)).

## What can be removed, and what cannot

- **Can remove from deployed runtime:** IRON DSL, MLIR-AIE compiler, Peano,
  `aiecc`, and compile-only design scripts, provided all required XDNA artifacts
  are built ahead of time and packaged with their manifest.
- **Can remove from runtime graph runner after a port:** imports of `aie.iron`,
  `aie.iron.device`, and `aie.utils.NPUKernel`.
- **Still required for direct XRT:** XRT userspace libraries, the XDNA driver,
  and a Python-compatible `pyxrt` binding. This host's supplied binding is for
  CPython 3.12.
- **Still required to change kernels or schedules:** an XDNA compiler/toolchain.
  Direct XRT only launches compiled artifacts; it does not replace MLIR-AIE
  code generation.

For a three-block linked artifact, direct XRT can launch the compiled stage as
one kernel and allocate only the stage input, packed weights, and stage output.
The FIFO links and tile schedule live inside the compiled image; they do not
need to be reconstructed on the host.

## Port plan and main risks

1. **Implemented:** `scripts/xdna/xdna_xrt_runtime.py` owns a process-wide
   device handle and bounded cache of artifact contexts/kernels, instruction
   BOs, and argument BOs. The RPC XRT path runs inside its serialized server
   process, so those caches survive between requests. The ResNet runner can
   select it for Conv, fused bottleneck, and MaxPool artifacts with the RPC
   option `runtime_backend: "xrt"`. Tensor
   BOs use each kernel argument's group ID and retain their host mappings.
   Residual Add+ReLU artifacts use the same backend. Immutable fused weights
   are uploaded once per XRT tensor, with a content digest so a later request
   using different model weights refreshes the BO.
2. The new `XRTTensor` exposes shape/dtype, writable
   host mapping, explicit upload/download, and the raw BO used for dispatch.
   Track dirty state so device-resident Q/DQ views do not trigger host copies.
3. The blocking kernel wrapper uses the existing IRON runtime ABI, validates
   tensor group IDs and launch argument positions, waits with a bounded timeout,
   and checks the final XRT command state. Reading kernel argument counts and
   dimensions from XCLBIN metadata is still follow-up work.
4. **RPC validation:** direct-XRT full graph output matched IRON exactly for
   the ResNet quicktest. One matched run measured 76.9 ms for XRT and 84.6 ms
   for IRON; later samples varied widely (82–180 ms), so this is not a stable
   performance claim. Three independently dispatched XRT bottlenecks also
   matched IRON at every captured block boundary. Their artifact argument
   groups differed between adjacent XCLBINs, so the runtime correctly falls
   back to host staging instead of passing an incompatible BO directly.
5. **RPC cache validation:** two separate requests in one server process
   reused the same 16 artifact contexts/kernels: the XRT kernel cache stayed
   at 16 misses while hits increased from 84 to 184. The fused block and graph
   outputs were identical across requests. This now follows Vitis AI's
   persistent runtime-context pattern, though runner metadata and some scratch
   buffers are still rebuilt per request.
6. **Workspace ownership:** scratch BOs are keyed by the exact loaded kernel
   context and compiled shape, not only tensor dimensions. This avoids passing
   a BO allocated from one artifact context to another kernel whose argument
   bank group differs. Context caching is capped at the device's 16-context
   limit; an unseen artifact evicts cached contexts before loading.
7. **Full-graph profiling:** the all-XRT ResNet quicktest completes with 16
   cached contexts and exact CPU-reference output, at 78.3 ms over two measured
   iterations (single-run sample). Routing convolutions with output M <= 64 to
   exact Torch int8 CPU gives 13.6 ms over three measured iterations, with
   zero output error. The uint8 MaxPool kernel reduced its own profiled time
   from 1.53 ms to 0.80 ms, but a paired full-graph sample was slower (16.2 ms
   vs. 13.6 ms); more stable repeated timing is needed before recommending it.
8. Next, reduce context switches and build a legal single-XCLBIN region for
   adjacent operators. The existing linked three-block experiment is still
   numerically incorrect; it remains excluded from inference and benchmarks.
9. Keep IRON as an optional backend until correctness, context eviction,
   timeout recovery, and output synchronization match. Then let RPC deployments
   choose either `iron` or `xrt` runtime without changing compile artifact
   formats.

An RPC comparison fused the three 8x8 layer1 bottlenecks and remained exactly
equal to the CPU reference, but averaged 107.8 ms versus 76.3 ms unfused. The
fused calls took 11.1 ms for the projection block and 6.9 ms each for the two
identity blocks. For this small feature map, graph fusion saves launches but
costs more than the separate Conv schedule; these blocks should stay opt-in
until the fused kernels are retiled or the runtime gains a lower-overhead
multi-kernel dispatch path.

The cross-XCLBIN buffer question has a local implementation precedent: the
installed IRON `XRTTensor` creates buffers with `pyxrt.bo(device, ...)` and its
runtime passes those device-owned BOs to each kernel context. A direct wrapper
can mirror this device-owned allocation model rather than allocating each BO
from an individual hardware context. We should still verify with two distinct
XCLBINs in the new wrapper before claiming device-resident handoffs are
equivalent; preserving the existing model is the lowest-risk starting point.

## What to take from Vitis AI

Vitis AI's useful precedent is its **subgraph runtime architecture**, not an
API that can launch our artifacts directly. Its ONNX Runtime EP partitions the
ONNX graph, compiles supported regions, and caches compiled context so later
sessions can load the result without recompiling. The public DynamicDispatch
project describes an operator library, graph metadata generation, a transaction
fuser, and a runtime that dispatches fused operators through an ONNX Runtime
custom op. This matches our goal of dispatching an XDNA region as one unit.

The installed Ryzen AI package contains `FusionSubgraphRuntime` and
`XRTMemoryAllocator` headers. They show a practical division worth copying:

- Keep per-subgraph XRT contexts, kernels, instruction BOs, and execution
  objects alive across calls.
- Represent graph inputs, outputs, scratch space, and constants separately.
- Reuse scratch and I/O allocations; keep constants immutable where possible.
- Pack operator tensors into known BO arguments and offsets, with metadata
  describing each operator's tensor arguments.
- Serialize access to shared input/output BOs, or allocate a separate slot per
  in-flight invocation.

For this repository, the corresponding artifact can remain our XDNA
`xclbin`/`insts.bin` pair plus a graph-region manifest. The RPC runner should
load one region runtime, retain its XRT objects and BOs, and submit a region
once per inference. The linked multi-block FIFO artifact is already shaped this
way at the host boundary, although its correctness issue still needs fixing.

Do **not** depend on Vitis AI's `FusionSubgraphRuntime` as the XDNA backend:
the installed runtime expects its own `DPU` kernel, operator metadata, and
transaction stream format. Our artifacts currently use IRON's `MLIR_AIE`
kernel and instruction stream ABI. Sharing a host architecture is practical;
sharing compiled kernels or binary formats is not established. A future custom
XDNA ONNX Runtime EP could use the same graph-partition/custom-op pattern, but
that is a larger integration than replacing `NPUKernel` in this RPC runner.

## RPC conversion timing

On the ResNet quicktest, a direct `compile_resnet_kernels.py` run took 67.7 s
cold and 39.1 s on a warmed repeat. The same model and compile options through
RPC took 39.9 s. The warmed pair differs by about 0.8 s (2%), suggesting RPC
staging is small compared with compilation; this is a single paired sample,
not a stable benchmark. Both paths compiled 17 artifacts (6 Conv and 11
operator kernels). The RPC path calls the same compiler script, so the cold
direct result should not be compared with the warmed RPC result as an RPC speed
gain.

The Vitis stack is observable locally under the Ryzen AI 1.8 environment:
`ryzenai_dynamic_dispatch/include/ryzenai/dynamic_dispatch/op_fuser/` contains
the subgraph runtime interface, and `memory_manager/` contains BO lifecycle
allocators. Public references: [Ryzen AI model compilation and deployment](https://ryzenai.docs.amd.com/en/1.7/modelrun.html) and [AMD DynamicDispatch](https://github.com/amd/DynamicDispatch).
