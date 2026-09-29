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
- Weight streaming was NOT the first bottleneck; scalar kernel gathers/epilogues were. New vectorized `--blocked` kernels (`kernels/fused_bottleneck_blocked.cc`, `blocked_stage.py`, runner `--fused-stage-blocked`) cover every ResNet-50 block shape (stride-2, 1x1..8x8 maps, multi-chunk weights, tap pruning) and are bit-exact vs ORT.
- Whole quicktest ResNet with four blocked stages (layer1..4, up to 8 blocks each): 11.6 ms avg on a loaded host, 0 CPU convs, logits identical to ORT CPU. Stage launches ~0.4 / 0.76 / 1.9 / 1.3 ms best-case; details in `docs/xdna-subgraph-dispatch.md`.
- Build/run on this host: `PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python LD_LIBRARY_PATH=/opt/xilinx/xrt/lib` with the IRON venv python; it has no torch/onnxruntime (make references with the Ryzen AI venv; the numpy CPU conv fallback is inexact for big layers). Set `aie::set_saturation(saturate)` in any new srs kernel, fully unroll MMUL accumulator groups, reserve static buffers with `Worker(data_size=...)`. The host is shared/loaded: compare artifacts with interleaved runs and take the minimum.
- Next (largest gaps to Vitis' ~1.6 ms): (1) merge the four launches / hand activations between stages on-device (per-launch ~0.3 ms setup + MaxPool 1.3 ms + stem conv/QDQ ~3 ms are now comparable to the body); (2) per-worker weight streams (today one broadcast FIFO per block feeds all four tiles, which each discard 3/4 of it; depth-1 so no compute/DMA overlap) and memtile weight staging to prefetch the next block; (3) wire `blocked` into the RPC `fused_stage` compile/run options and add a content-addressed artifact cache.

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
