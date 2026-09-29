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

## Linked stage fix (latest)

- The linked multi-block stage bug was a Python late-binding closure in `linked_bottleneck_stage_design.py` (all blocks used the last block's chunk counts); fixed via `_block_workers`. Linked layer1 stages of 2 and 3 blocks are now bit-exact; 3-block stage = 16.7 ms vs 24.9 ms for three separate fused blocks. See `docs/xdna-subgraph-dispatch.md`.
- To build/run on this host: `PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python LD_LIBRARY_PATH=/opt/xilinx/xrt/lib` with the IRON venv python; the IRON venv has no torch/onnxruntime (use `--cpu-backend numpy`, compute references with the Ryzen AI venv).
- Next: cut weight-streaming cost (167 KB per launch, one awaited chunk at a time) -- resident weights or un-awaited fills -- then extend linked stages to layer2-4.

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
