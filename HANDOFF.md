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

## Linked stage (latest)

- Bug fixed: Python late-binding closure in `linked_bottleneck_stage_design.py` made all blocks use the last block's chunk counts (`_block_workers`).
- Weight streaming is NOT the bottleneck (~1 ms for a 3-block stage); scalar kernel gathers/epilogues were (~15 of 16.7 ms). New vectorized `--blocked` kernels (`kernels/fused_bottleneck_blocked.cc`, `blocked_stage.py`, runner `--fused-stage-blocked`) are bit-exact and take the linked layer1 stage from 16.7 ms to 1.65 ms. See `docs/xdna-subgraph-dispatch.md`.
- Build/run on this host: `PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python LD_LIBRARY_PATH=/opt/xilinx/xrt/lib` with the IRON venv python; it has no torch/onnxruntime (use `--cpu-backend numpy`; make references with the Ryzen AI venv). Set `aie::set_saturation(saturate)` explicitly in any new srs kernel.
- Next: generalize the blocked kernels to layer2-4 (stride-2 first blocks, W<8 maps via flattened padded tiling, multi-chunk weights); then the remaining CPU convs (42 in the quicktest graph) are the main end-to-end cost (numpy CPU conv 30 ms; torch-int8 13.6 ms). Wire `blocked` into the RPC `fused_stage` compile/run options.

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
