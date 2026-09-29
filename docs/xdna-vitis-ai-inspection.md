# How Vitis AI / Ryzen AI 1.8 runs the quicktest ResNet-50 on Strix Halo XDNA2 (~1.55 ms)

Scratch/evidence lives in /tmp/vitis-inspect/ (strace.txt, shim.log, cache/r50/{compiled.*.xmodel,context.json},
mc_code.dec = the control-code ELF, hsi.json, kids.json = per-layer cost model, tl/record_timer_ts.json, xb/*.xclbin).

## 0. Caveats that matter before copying anything
* The quicktest model is **ResNet-50 at 1x3x32x32 input, 10 classes** (QDQ uint8 act / int8 weights, 386 nodes,
  53 conv + 2 gemm). Compiler `workload` = 168 M ops. `workload_on_arch` = 2.86 G (array ~6% utilised).
  So this is a **pure weight-streaming benchmark**: 25.3 MB of weights per inference, tiny activations (<=32 KB spill offsets).
* CNN path is NOT DynamicDispatch (that is the LLM/`dyn_bins` op-by-op path). It is the VAIML/"X2" flow:
  `compile_pass_manager: target AMD_AIE2P_4x8_CMC_Overlay, enable_txn_elf:1, vaiml_compile_x2_v2 ~750 ms, Graph Engine Runner 0.2`,
  run by libgraph-engine / flexmlrt (`flexmlrt::cert_xrt`, `ryzen_ai_xrt`).
* Working run recipe (system XRT 2.25 must be paired with ORT's venv): source /opt/xilinx/xrt/setup.sh; source venv; export
  RYZEN_AI_INSTALLATION_PATH=<venv>; LD_PRELOAD=/opt/xilinx/xrt/lib/libxrt_coreutil.so.2. (env in /tmp/vitis-inspect/env.sh; without it: "XRT is not installed").
* Measured today (loaded host, flock): 500 runs min **1.553 ms**, p10 1.596, median 1.614 (includes CPU Quantize ~25-90 us
  + Dequantize ~6-12 us + ORT overhead). ORT profile: the single `vitis_ai_ep_1` node = 1.58 ms.

## 1. Binary/overlay, contexts, launches (strace of ioctls + shim on amdxdna ioctls)
* No .xclbin/.txn file is opened at run time; everything is in-memory. Compile artifacts (with provider options
  `cache_dir, cache_key, enable_cache_file_io_in_mem=0`): `compiled.AMD_AIE2P_4x8_CMC_Overlay.xmodel` (26.1 MB) + context.json.
* The xmodel embeds an **xclbin (180,890 B)**: a *generic unified overlay* with a dummy `vadd` kernel, AIE_PARTITION column_width=8,
  start col 0, **PDI only 18.9 KB** (same family as flexml/.../stx/unified-2x4x4.xclbin 111,919 B; also 4-col 92 KB and llm 115 KB variants
  ship inside libvaiml.so). The xclbin holds no model logic; it is not per-shape.
* **One HW context per session**: 1x CREATE_HWCTX (num_tiles=32 = 8 cols x 4 cores, max_opc=2048), 1x CONFIG_HWCTX;
  xrt-smi shows Partition columns [0..7], 1 context. "PDI Swap times: 0". No xclbin/context switching at all.
* **One EXEC_CMD (+ one SYNCOBJ wait) per inference** (23 exec for 20 warmup + 3 runs; args: 4 BOs, cmd BO handle 21).
* What actually drives the array is an **ELF control program** (`mc_code`, zlib inside the xmodel, 1,296,504 B ELF): sections
  `.ctrldata` 0x116b80 = 1,141,632 B, `.ctrltext` 125,344 B, `.preempt_save` 4,188, `.preempt_restore` 4,572. The BO list created at init
  matches exactly: 8 x 1,141,632 B CMD BOs (**one ctrldata copy per column**), BOs of 125,344 / 4,188 / 4,572 B, a 4 MB + 4 KB
  CMD BO (preemption/scratch), 64 MB DEV_HEAP. So: uC-in-each-shim-column (CERT) executes a fixed 125 KB program that
  walks 71 layer descriptors in the 1.14 MB per-column data (BD/DMA descriptors, ~16 KB per layer per column). The host does not
  sequence layers; one launch runs the whole graph including the 71 layer-to-layer syncs.
  Preemption is enabled (enable_preemption=1) but costs nothing measurable here.

## 2. Weights
* One host SHMEM BO `REG_0` = **25,326,400 B** ("CONST", `AssignMode::LINEAR_NO_REUSE`) holding *all* layers' weights+bias+requant params,
  layer offsets in hsi.json (`weights.layers`, 71 entries). ONNX int8 weight tensors total 23,586,624 B, so the packing is ~7% overhead:
  **raw int8, no compression, no BFP**; weight layout is OHWI-style (`onnx::Conv_570_quantized shape [512,1,1,128]` for a 1x1 conv, size = exactly
  prod(shape)). Per-op kernel params (32 x u32 `param_hex` + 128 B `param_buffer`, shifts/zp/c0-c3 etc.) sit at the tail of REG_0 (`kernel_param_addr=25324736`).
* Other buffers: REG_1 170,000 B (IFM/OFM/INTER ddr buffers, activation spills, max offset 32 KB), REG_2 13,312 B, REG_3 2,144 B.
* **Weights are NOT resident**: 25 MB >> memtile 4 MB + L1 2 MB, and compiler options say `enable_weights_stationary=0, enable_weights_prefetch=0`.
  They are re-streamed DDR->memtile->cores every inference. There is no cross-inference weight caching. The trick is not less traffic, it is more parallel traffic (below).
* Activations also go layer-to-layer through DDR (every op `input/output_memory_status=ON_DDR`; `enable_mt_fusion=0`), but at 32x32 they are only KBs.

## 3. Graph compile/fusion
* Not block-fused. The xmodel has 76 child "subgraphs" = 71 executed layers: 55 qlinear-conv2d (fc gemm rewritten to conv, "TransferQDQMatMulToConv2d"),
  16 qlinear-eltwise (residual Add, separate op ~1.6 us model each), maxpool, transposes/upload/download; ReLU/requant fused in the conv, DQ/Q at the ends run on CPU.
* **Every layer uses all 8 columns** (`enable_col_num=8` on all 76 children) and layers run strictly one after another. Each layer picks a tiling mode
  (`OH4OC8` for most convs, `OH8OC4`, `OH16OC2` for the 3x3 with 1x1 spatial map in layer4): the output-channel dimension is split across the columns
  (OC8 = 8-way) so each column's shim streams only 1/8 of that layer's weights, in parallel.
* Compiler settings of interest (xmodel root attrs): tiling_algorithm=high_throughput, prefetch_lp=8, prefetch_lp_reorder_load=1,
  prefetch_lp_warm_start=1, mergesync_column_as_all=1, enable_x2_ge=1, engine_parallel=1, enable_ddr_dominator=1, enable_control_optimization=1,
  mt_block_mode=1, force_mode=65536, set_conv_aie_mode=3, set_dwconv_aie_mode=3, enable_matmul_to_conv=1.
* The compiler carries a per-layer cost model (kids.json): `ddr_load_cycle`, `mt2aie_cycle`, `kernel_computation(cycle)`, `ddr_save_cycle`,
  `projection_bottleneck`. Sum of model latency = 1.315 M cycles (~0.73 ms at 1.8 GHz). Bottleneck per layer is LOAD (ddr) or MT2AIE (memtile->core weights),
  never kernel compute (kernel cycles are 5-20% of load cycles). Sum(ddr_load_cycle)=773 k cycles for 25.3 MB = 32.7 B/cycle = **~8 shim streams x 4 B/cycle (~59 GB/s @1.8 GHz)**.
  Example layer4.0.conv2 (2.36 MB): ddr_load 85,440 cyc; kernel 34,224 cyc; mt2aie 177,536 cyc (memtile->core is the limit there).

## 4. Profiling/timings we can extract
* xrt.ini `[Debug] ml_timeline=true` (in cwd) produces record_timer_ts.json: 2 timestamps per inference (before/after the transaction);
  device time min **2,697,674 timer cycles** (median 2,704,054) vs host gap between runs 48 k cycles (~27 us): the NPU is busy ~98% of the wall time
  (timer ~1.75 GHz => ~1.5 ms). No per-layer hardware timestamps are emitted (only ids 0/1). `aie_record_timer` metadata confirms the two-timestamp scheme.
* Real time is ~2.05x the compiler's own cost model (2.70 M vs 1.32 M cycles): even Vitis reaches only ~17 GB/s effective (25.3 MB / 1.5 ms)
  against a ~59 GB/s model / 8x7.2 GB/s shim ceiling; the remainder is per-layer sync/setup (71 layers -> ~10-15 us overhead each) + imperfect prefetch overlap.
* ORT profile (VAIP_...): single node vitis_ai_ep_1, 1.58-1.87 ms; per-op ORT data unavailable (fused into one custom op).
* Per-layer model costs top out at: layer4.0.conv2 107 us, layer4.1/2.conv2 91 us each, layer4.0.downsample 39 us, layer3.0.conv2 23 us, layer4.x.conv1 24 us => layer4 ~ 45% of the model total.

## 5. Actionable takeaways (ranked by expected payoff)
1. **Use all 8 shim weight streams for every layer/block instead of one per active block kind.** Evidence: Vitis splits each layer's output channels
   over all 8 columns (OC8/OH4OC8 tiling, enable_col_num=8 on every layer) and its cost model assumes ~32 B/cycle DDR read; shim MM2S = 4 B/cycle/channel = the ~7 GB/s per stream
   we measure. Our body xclbin = 21 MB over 1-2 concurrent streams -> 3.5 ms; the same 21 MB over 8 streams is a 0.4 ms floor, over 4 streams ~0.75 ms.
   Design: weight-split (OC) each conv over the 8 columns and run blocks sequentially (block 1 ... 16), not "one column per kind". Kind-per-column is exactly the wrong axis for a
   weight-bound net: 7 of 8 columns idle their weight DMA while one kind runs. (If keeping kind-per-column, at least make consecutive blocks run in a software-pipelined way so >= 4-8 kinds stream at once.)
2. **Push weight packing into one contiguous DDR BO with per-layer offsets + a tiny per-layer param block** (Vitis: one 25.3 MB BO, raw int8, params at tail, 71 layer offsets).
   No compression/BFP was used, so no need to invent a format; the win is in stream parallelism, not bytes. Keep fixed generic kernels + runtime parameter words (they do the same as our slot-header shifts).
3. **Single context, single launch, controller-sequenced layers.** Vitis: 1 xclbin (generic, 18.9 KB PDI), 1 HWCTX, 1 EXEC_CMD, 0 PDI swaps; 71 layers stepped by a 125 KB ctrl program
   with 8 per-column 1.14 MB descriptor tables. We already have the one-xclbin/one-launch property; keep it and avoid any per-layer host round trips. Their loss to sync is ~10-15 us/layer, so
   fusing the three convs of a bottleneck into a layer-group with one sync (as we do) is a potential *win* over Vitis if the weight streams are parallelised.
4. **Double-buffer/prefetch the next layer's weights while the current computes** (compiler flags prefetch_lp=8, prefetch_lp_reorder_load, warm_start; cost model overlaps DDR->memtile with
   memtile->core). Their measured 2x gap to the model shows prefetch/overlap is imperfect even for them; our memtile weight FIFO depth already tries this - measure it per layer.
5. **Layer4 dominates** (3x3 conv on a 1x1 map with 2.36 MB weights; memtile->core bound, tile mode OH16OC2 = mostly OC split). Give layer4 the most streams and keep
   memtile->core transfers broadcast-free (each core its own OC slice) ; expect ~45% of total time here.
6. **Don't chase weight residency**: 25 MB cannot live in 4 MB memtile+2 MB L1; Vitis does not try (weights_stationary=0). Also no evidence of int4/BFP for this CNN; a8w8 raw.
7. Low priority: Vitis keeps activations in DDR between layers (ON_DDR) at 32x32 because they are ~KB; this costs almost nothing -> our DDR round-trip of activations is not the bottleneck.
   Q/DQ (25-90 us) and the ORT wrapper (~30 us host gap) are a fixed 5-8% of their 1.55 ms; pre-quantized input removes it.

## Confidence / open items
* Certain: 1 xclbin, 1 hwctx (32 tiles, 8 cols), 1 exec per inference, 25.3 MB raw weight BO, ctrl-ELF structure and BO sizes, compiler options, cost-model numbers, device-time measurement.
* Inferred (not directly observed on the wire): weights are split by OC across columns (from tiling-mode names OH4OC8/OC8, enable_col_num=8 and the 8 ctrldata copies);
  shim rate 4 B/cycle (matches our measured 7 GB/s). To confirm, decode .ctrldata BDs in mc_code.dec (aiebu-dump could not parse it; sections start at file offset 0xe0/0x116c60).
* Not run: DynamicDispatch/`dyn_bins` path (irrelevant to CNN here); no public-doc fetch was needed.
