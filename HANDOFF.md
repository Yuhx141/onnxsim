# XDNA ResNet RPC handoff

## Current state

- Branch: `codex/xdna-resnet-graph-fusion` (upstream: `origin/codex/xdna-resnet-graph-fusion`).
- Device runs and compiles were performed through the RPC server; no RPC server is left running.
- This note summarizes the XDNA/RPC progress on this branch.
- Do not add `.commandcode/`; it is unrelated workspace content.

## Runtime findings

- The direct XRT runtime is opt-in with RPC `runtime_backend: "xrt"` and uses persistent XRT contexts in the serialized RPC process.
- XRT context cache is capped at 16 on this device. Kernel workspaces must be tied to their owning kernel/context and shape; XRT argument bank groups can differ across compiled artifacts.
- All-XRT quicktest ResNet output matched the Torch CPU reference exactly. Recent all-XRT samples were around 76–78 ms.
- Best measured quicktest schedule routes Conv output-M <= 64 to exact Torch int8 CPU while retaining stem Conv, MaxPool, and one native quantized Add+ReLU on XDNA: 13.6 ms average over 3 measured iterations, zero output error. This is a hybrid schedule, not full-device execution.
- Fusing `/layer1/layer1.0`–`.2` is exact but slower on this 8x8 feature map: 107.8 ms vs 76.3 ms unfused. Per-block fused times were 11.1 ms (projection) and 6.9 ms (identity). Keep those fusions opt-in pending retile/runtime work.
- uint8 MaxPool reduced kernel time from about 1.53 ms to 0.80 ms, but one full-graph sample regressed; repeat end-to-end timing before recommending it.

## XDNA body status (latest)

- All 16 ResNet-50 bottlenecks run in ONE xclbin (`resnet_body_design.py`: one core column per block kind, same-shaped blocks iterate on it with runtime shifts from a slot header). Bit-exact vs ORT, 3.7 ms per launch. Full graph with host stem Conv + MaxPool: ~7-8 ms on this (heavily loaded) host, logits identical to ORT CPU. See `docs/xdna-subgraph-dispatch.md` and `docs/rpc.md` (`kind="resnet_body"`, `fused_body`, `host_maxpool`).
- Key findings: xclbin/PDI switches cost ~0.75 ms+ each (multi-device full ELF pays the same), so avoid reconfiguration; weight streaming (~7 GB/s per stream) is now the body's floor (~3 ms of 3.7 ms); scalar-loop kernels, un-unrolled accumulator groups, and runtime constexpr searches each cost 2-10x.
- Build/run on this host: `PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python LD_LIBRARY_PATH=/opt/xilinx/xrt/lib` with the IRON venv python (no torch/onnxruntime there; make references with the Ryzen AI venv; `uvx ruff` for formatting). Set `aie::set_saturation(saturate)` in new srs kernels, fully unroll MMUL accumulator groups, reserve static buffers with `Worker(data_size=...)`. Compare artifacts with interleaved runs and take the minimum: the host load average is >10.
- Next: (1) more weight bandwidth for the active group (memtile staging/prefetch, 2 streams per group by sharing input channels); (2) move stem Conv/MaxPool onto spare NPU cores of the body xclbin instead of the host; (3) cache the compile (content-addressed) and run device tests in CI-less scripts; (4) tests that exercise `resnet_body` through the RPC server on the device host.

## Compile timing

For `test_model.onnx`, with `npu2`, 8 columns, and `optimize_small_m` enabled:

- Direct compile: 67.7 s cold, 39.1 s warmed.
- RPC compile: 39.9 s in one warmed comparison.
- Both produced 17 artifacts (6 Conv and 11 operator kernels). The warmed RPC/direct difference was about 0.8 s (2%), one paired sample only. The RPC path runs the same compile script; do not treat the cold/warm difference as an RPC speedup.

## Follow-up

1. Keep comparing optimizations over repeated RPC runs with correctness enabled.
2. Improve fused bottleneck kernel tiling and reduce XRT launch overhead before enabling small-spatial fusion.
3. Investigate generic device-resident activation/QDQ/epilogue handoffs; current Conv path still stages most boundaries through host computation.
4. Consider a content-addressed artifact cache for repeated RPC compiles, with robust invalidation for model, compiler sources, options, and toolchain.

See `docs/xdna-runtime-without-iron.md` and `docs/xdna-subgraph-dispatch.md` for design and profile details.
