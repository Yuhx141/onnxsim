/* From-scratch native ARM64 FastRPC client: proves TVM's Python/C++ RPC stack is NOT required
 * to drive a custom Hexagon skel. Talks to mini_rpc.so (our own, non-TVM skel, built by
 * ourselves via qaic) purely through libcdsprpc.so's remote_handle64_open/invoke/close, using
 * the qaic-generated stub (mini_rpc_stub.c) for marshaling. No tvm.rpc, no tvm.contrib.hexagon,
 * no MinRPC, no libhexagon_rpc_skel.so anywhere in this binary or its dependency graph. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mini_rpc.h"
#include <remote.h>

#define GEMM_A_LEN 3481600
#define GEMM_B_LEN 16384
#define GEMM_C_LEN 55705600
#define GEMM_A_PATH "/data/local/tmp/native_transport/gemm_a.bin"
#define GEMM_B_PATH "/data/local/tmp/native_transport/gemm_bp.bin"
#define GEMM_C_PATH "/data/local/tmp/native_transport/gemm_c_out.bin"

#define REQUANT_A_LEN 8000000
#define REQUANT_C_LEN 2000000
#define REQUANT_A_PATH "/data/local/tmp/native_transport/requant_a.bin"
#define REQUANT_C_PATH "/data/local/tmp/native_transport/requant_c_out.bin"

#define SUBGRAPH_A_LEN 3527056
#define SUBGRAPH_C_LEN 3481600
#define SUBGRAPH_A_PATH "/data/local/tmp/native_transport/subgraph_image_padded.bin"
#define SUBGRAPH_C_PATH "/data/local/tmp/native_transport/subgraph_maxpool_out.bin"

#define SMALL_SUBGRAPH_A_LEN 5776
#define SMALL_SUBGRAPH_C_LEN 4096
#define SMALL_SUBGRAPH_A_PATH "/data/local/tmp/native_transport/small_image_padded.bin"
#define SMALL_SUBGRAPH_C_PATH "/data/local/tmp/native_transport/small_maxpool_out.bin"

#define BLOCK1_A_LEN 3481600
#define BLOCK1_C_LEN 13926400
#define BLOCK1_A_PATH "/data/local/tmp/native_transport/block1_input_packed.bin"
#define BLOCK1_C_PATH "/data/local/tmp/native_transport/block1_output.bin"

#define BOXHEAD_A_LEN 3211264
#define BOXHEAD_B_LEN 25690112
#define BOXHEAD_C_LEN 131072
#define BOXHEAD_A_PATH "/data/local/tmp/native_transport/boxhead_a.bin"
#define BOXHEAD_B_PATH "/data/local/tmp/native_transport/boxhead_bp.bin"
#define BOXHEAD_C_PATH "/data/local/tmp/native_transport/boxhead_c_out.bin"

static unsigned char* read_file_exact(const char* path, int expect_len) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  unsigned char* buf = malloc(expect_len);
  size_t n = fread(buf, 1, expect_len, f);
  fclose(f);
  if ((int)n != expect_len) { free(buf); return NULL; }
  return buf;
}

int main(int argc, char** argv) {
  const char* uri = argv[1];  /* e.g. "file:///data/local/tmp/native_transport/mini_rpc.so?mini_rpc_skel_handle_invoke&_dom=cdsp" */

  /* SDK default is only 16 KiB per FastRPC DSP thread. Generated call 61 needs a 0x5798-byte
   * frame by itself, so configure the cDSP worker stack before making any other RPC call. */
  struct remote_rpc_thread_params thread_params;
  thread_params.domain = CDSP_DOMAIN_ID;
  thread_params.prio = -1;
  thread_params.stack_size = 64 * 1024;
  int trc = remote_session_control(FASTRPC_THREAD_PARAMS, &thread_params, sizeof(thread_params));
  printf("set cDSP FastRPC stack=65536 rc=%d\n", trc);
  if (trc != 0) return 2;

  /* The missing piece: unsigned .so loading on the CDSP is refused (AEE_ECONNREFUSED) unless
   * this process explicitly opts in per-session first. TVM's own launcher/session code
   * (apps/hexagon_launcher/launcher_android.cc, src/runtime/hexagon/rpc/android/session.cc)
   * does exactly this before its first hexagon_rpc_open() call. */
  struct remote_rpc_control_unsigned_module unsigned_pd;
  unsigned_pd.domain = CDSP_DOMAIN_ID;
  unsigned_pd.enable = 1;
  int urc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_pd, sizeof(unsigned_pd));
  printf("enable_unsigned_pd rc=%d\n", urc);

  remote_handle64 h = 0;
  int rc = mini_rpc_open(uri, &h);
  if (rc != 0) {
    printf("open failed rc=%d\n", rc);
    return 1;
  }
  printf("open OK, handle=0x%llx\n", (unsigned long long)h);

  int32_t result = -1;
  rc = mini_rpc_run_add(h, 40, 2, &result);
  printf("run_add(40,2) rc=%d result=%d\n", rc, result);

  unsigned char a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  unsigned char b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  unsigned char c[8] = {0};
  rc = mini_rpc_run_kernel(h, a, 8, b, 8, c, 8);
  printf("run_kernel (placeholder byte-add) rc=%d out=", rc);
  for (int i = 0; i < 8; i++) printf("%d ", c[i]);
  printf("\n");

#ifdef TG_OPENPILOT_PROBE
  #ifndef TG_PROBE_INPUT_BYTES
  #error "TG_PROBE_INPUT_BYTES must be set for the openpilot model-kernel probe"
  #endif
  #ifndef TG_PROBE_OUTPUT_BYTES
  #error "TG_PROBE_OUTPUT_BYTES must be set for the openpilot model-kernel probe"
  #endif
  #ifndef TG_PROBE_REPEATS
  #define TG_PROBE_REPEATS 5
  #endif
  const char* probe_input_path = "/data/local/tmp/native_transport/openpilot_probe_input.bin";
  const char* probe_output_path = "/data/local/tmp/native_transport/openpilot_probe_output.bin";
  unsigned char* probe_input = read_file_exact(probe_input_path, TG_PROBE_INPUT_BYTES);
  if (!probe_input) {
    fprintf(stderr, "probe input missing or wrong size: %s\n", probe_input_path);
    mini_rpc_close(h);
    return 3;
  }
  unsigned char* probe_output = malloc(TG_PROBE_OUTPUT_BYTES);
  if (!probe_output) { free(probe_input); mini_rpc_close(h); return 4; }
  double probe_times[TG_PROBE_REPEATS];
  int probe_rc = 0, probe_count = 0;
  for (int probe_i = 0; probe_i < TG_PROBE_REPEATS; probe_i++) {
    struct timespec probe_t0, probe_t1;
    clock_gettime(CLOCK_MONOTONIC, &probe_t0);
    probe_rc = mini_rpc_run_kernel(h, probe_input, TG_PROBE_INPUT_BYTES, probe_input, 0,
                                   probe_output, TG_PROBE_OUTPUT_BYTES);
    clock_gettime(CLOCK_MONOTONIC, &probe_t1);
    probe_times[probe_i] = (probe_t1.tv_sec - probe_t0.tv_sec) * 1000.0 +
                           (probe_t1.tv_nsec - probe_t0.tv_nsec) / 1e6;
    probe_count++;
    if (probe_rc != 0) break;
  }
  for (int i = 1; i < probe_count; i++) {
    double value = probe_times[i]; int j = i - 1;
    while (j >= 0 && probe_times[j] > value) { probe_times[j + 1] = probe_times[j]; j--; }
    probe_times[j + 1] = value;
  }
  printf("openpilot kernel probe rc=%d repeats=%d median_wall_ms=%.3f\n", probe_rc,
         probe_count, probe_times[probe_count / 2]);
  FILE* probe_file = fopen(probe_output_path, "wb");
  if (!probe_file) { free(probe_input); free(probe_output); mini_rpc_close(h); return 5; }
  fwrite(probe_output, 1, TG_PROBE_OUTPUT_BYTES, probe_file);
  fclose(probe_file);
  printf("wrote %s (%d bytes)\n", probe_output_path, TG_PROBE_OUTPUT_BYTES);
  free(probe_input);
  free(probe_output);
  mini_rpc_close(h);
  return probe_rc;
#endif

#ifdef TG_OPENPILOT_GRAPH
  #ifndef TG_GRAPH_INPUT_BYTES
  #error "TG_GRAPH_INPUT_BYTES must be set for the openpilot graph runner"
  #endif
  #ifndef TG_GRAPH_OUTPUT_BYTES
  #error "TG_GRAPH_OUTPUT_BYTES must be set for the openpilot graph runner"
  #endif
  #ifndef TG_GRAPH_WEIGHT_BYTES
  #error "TG_GRAPH_WEIGHT_BYTES must be set for the openpilot graph runner"
  #endif
  #ifndef TG_GRAPH_WEIGHT_CHUNK
  #define TG_GRAPH_WEIGHT_CHUNK 8388608
  #endif
  const char* graph_weights_path = "/data/local/tmp/native_transport/openpilot_graph_weights.bin";
  const char* graph_input_path = "/data/local/tmp/native_transport/openpilot_graph_input.bin";
  const char* graph_expected_path = "/data/local/tmp/native_transport/openpilot_graph_expected.bin";
  const char* graph_output_path = "/data/local/tmp/native_transport/openpilot_graph_output.bin";
  unsigned char* graph_weights = read_file_exact(graph_weights_path, TG_GRAPH_WEIGHT_BYTES);
  unsigned char* graph_input = read_file_exact(graph_input_path, TG_GRAPH_INPUT_BYTES);
  unsigned char* graph_expected = read_file_exact(graph_expected_path, TG_GRAPH_OUTPUT_BYTES);
  unsigned char* graph_output = malloc(TG_GRAPH_OUTPUT_BYTES);
  if (!graph_weights || !graph_input || !graph_expected || !graph_output) {
    fprintf(stderr, "openpilot graph files missing or have wrong sizes\n");
    free(graph_weights); free(graph_input); free(graph_expected); free(graph_output);
    mini_rpc_close(h);
    return 6;
  }
  printf("loading %d openpilot weight bytes in %d-byte FastRPC chunks...\n",
         TG_GRAPH_WEIGHT_BYTES, TG_GRAPH_WEIGHT_CHUNK);
  int graph_rc = 0;
  unsigned char graph_empty[1] = {0};
  for (int offset = 0; offset < TG_GRAPH_WEIGHT_BYTES; offset += TG_GRAPH_WEIGHT_CHUNK) {
    int chunk = TG_GRAPH_WEIGHT_BYTES - offset;
    if (chunk > TG_GRAPH_WEIGHT_CHUNK) chunk = TG_GRAPH_WEIGHT_CHUNK;
    graph_rc = mini_rpc_run_kernel(h, graph_weights + offset, chunk,
                                   (const unsigned char*)&offset, sizeof(offset), graph_empty, 0);
    if (graph_rc != 0) {
      fprintf(stderr, "load weights failed at offset %d size %d rc=%d\n", offset, chunk, graph_rc);
      break;
    }
  }
  if (graph_rc == 0) {
    struct timespec graph_t0, graph_t1;
    clock_gettime(CLOCK_MONOTONIC, &graph_t0);
    graph_rc = mini_rpc_run_kernel(h, graph_input, TG_GRAPH_INPUT_BYTES, graph_empty, 0, graph_empty, 0);
    int32_t batch[2];
    const int batch_calls = TG_GRAPH_BATCH_CALLS;
    for (int start = 0; graph_rc == 0 && start < TG_GRAPH_CALLS; start += batch_calls) {
      batch[0] = start;
      batch[1] = (start + batch_calls <= TG_GRAPH_CALLS) ? batch_calls : TG_GRAPH_CALLS - start;
      struct timespec batch_t0, batch_t1;
      clock_gettime(CLOCK_MONOTONIC, &batch_t0);
      graph_rc = mini_rpc_run_kernel(h, graph_empty, 0, (const unsigned char*)batch,
                                     sizeof(batch), graph_empty, 0);
      clock_gettime(CLOCK_MONOTONIC, &batch_t1);
      double batch_ms = (batch_t1.tv_sec - batch_t0.tv_sec) * 1000.0 +
                        (batch_t1.tv_nsec - batch_t0.tv_nsec) / 1e6;
      printf("openpilot batch start=%d count=%d rc=%d wall_ms=%.3f\n",
             start, batch[1], graph_rc, batch_ms);
      if (graph_rc != 0) fprintf(stderr, "openpilot graph batch %d rc=%d\n", start, graph_rc);
    }
    if (graph_rc == 0)
      graph_rc = mini_rpc_run_kernel(h, graph_empty, 0, graph_empty, 0,
                                     graph_output, TG_GRAPH_OUTPUT_BYTES);
    clock_gettime(CLOCK_MONOTONIC, &graph_t1);
    double graph_ms = (graph_t1.tv_sec - graph_t0.tv_sec) * 1000.0 +
                      (graph_t1.tv_nsec - graph_t0.tv_nsec) / 1e6;
    printf("openpilot graph rc=%d wall_ms=%.3f\n", graph_rc, graph_ms);
  }
  if (graph_rc == 0) {
    FILE* graph_file = fopen(graph_output_path, "wb");
    if (!graph_file || fwrite(graph_output, 1, TG_GRAPH_OUTPUT_BYTES, graph_file) != TG_GRAPH_OUTPUT_BYTES) {
      if (graph_file) fclose(graph_file);
      fprintf(stderr, "could not write graph output\n");
      graph_rc = -1;
    } else fclose(graph_file);
    int graph_match = memcmp(graph_output, graph_expected, TG_GRAPH_OUTPUT_BYTES) == 0;
    printf("full graph bit-exact to QEMU zero-input reference: %s\n", graph_match ? "yes" : "no");
    if (!graph_match) graph_rc = -1;
  }
  free(graph_weights); free(graph_input); free(graph_expected); free(graph_output);
  mini_rpc_close(h);
  return graph_rc == 0 ? 0 : 7;
#endif

  /* Real hex_gemm_kernel.py-generated vrmpy GEMM at the flagship Mask R-CNN backbone shape
   * (cin=64,cout=256,m=54400) -- only runs if gen_gemm_test_data.py's output files were pushed
   * to the device first; skipped gracefully otherwise so the base PoC still works standalone. */
  unsigned char* ga = read_file_exact(GEMM_A_PATH, GEMM_A_LEN);
  unsigned char* gb = read_file_exact(GEMM_B_PATH, GEMM_B_LEN);
  if (ga && gb) {
    unsigned char* gc = malloc(GEMM_C_LEN);
    printf("running real hex_gemm kernel (cin=64,cout=256,m=54400)...\n");
    int grc = mini_rpc_run_kernel(h, ga, GEMM_A_LEN, gb, GEMM_B_LEN, gc, GEMM_C_LEN);
    printf("run_gemm_kernel rc=%d\n", grc);
    if (grc == 0) {
      FILE* out = fopen(GEMM_C_PATH, "wb");
      fwrite(gc, 1, GEMM_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against a numpy reference host-side\n", GEMM_C_PATH, GEMM_C_LEN);
    }
    free(gc);
  } else {
    printf("gemm test data not found at %s / %s, skipping real-kernel test\n", GEMM_A_PATH, GEMM_B_PATH);
  }
  free(ga);
  free(gb);

  /* Real hex_requantize_kernel.py-generated requantize, n=2,000,000 (a real-scale, RPC-safe
   * slice of the stem conv's output) -- only runs if gen_requantize_test_data.py's output was
   * pushed first; skipped gracefully otherwise. */
  unsigned char* ra = read_file_exact(REQUANT_A_PATH, REQUANT_A_LEN);
  if (ra) {
    unsigned char* rc_out = malloc(REQUANT_C_LEN);
    printf("running real hex_requantize kernel (n=2000000)...\n");
    int rrc = mini_rpc_run_kernel(h, ra, REQUANT_A_LEN, ra, 0, rc_out, REQUANT_C_LEN);
    printf("run_requantize_kernel rc=%d\n", rrc);
    if (rrc == 0) {
      FILE* out = fopen(REQUANT_C_PATH, "wb");
      fwrite(rc_out, 1, REQUANT_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against the numpy reference host-side\n", REQUANT_C_PATH, REQUANT_C_LEN);
    }
    free(rc_out);
  } else {
    printf("requantize test data not found at %s, skipping\n", REQUANT_A_PATH);
  }
  free(ra);

  /* Stage 2: the fused stem7x7 -> bias_add -> requantize -> layout_transform -> maxpool
   * subgraph, real backbone.onnx weights/bias/scales, real quantized+padded image in -- only
   * runs if the image was pushed first; skipped gracefully otherwise. Times the on-device call
   * (the only thing crossing the RPC boundary is this input and the final maxpool output --
   * every intermediate activation stays on-device, see subgraph_driver.c). */
  /* Diagnostic: a tiny-scale (ih=iw=32) copy of the same pipeline, to isolate whether a
   * full-scale failure is a real per-process memory-size limit vs. a logic bug -- run before
   * the full-scale test so its result is available even if the full-scale one crashes. */
  unsigned char* ssa = read_file_exact(SMALL_SUBGRAPH_A_PATH, SMALL_SUBGRAPH_A_LEN);
  if (ssa) {
    unsigned char* ssc = malloc(SMALL_SUBGRAPH_C_LEN);
    printf("running SMALL-scale (32x32) subgraph diagnostic...\n");
    int ssrc = mini_rpc_run_kernel(h, ssa, SMALL_SUBGRAPH_A_LEN, ssa, 0, ssc, SMALL_SUBGRAPH_C_LEN);
    printf("run_small_subgraph rc=%d\n", ssrc);
    if (ssrc == 0) {
      FILE* out = fopen(SMALL_SUBGRAPH_C_PATH, "wb");
      fwrite(ssc, 1, SMALL_SUBGRAPH_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes)\n", SMALL_SUBGRAPH_C_PATH, SMALL_SUBGRAPH_C_LEN);
    }
    free(ssc);
  } else {
    printf("small subgraph test data not found at %s, skipping\n", SMALL_SUBGRAPH_A_PATH);
  }
  free(ssa);

  unsigned char* sa = read_file_exact(SUBGRAPH_A_PATH, SUBGRAPH_A_LEN);
  if (sa) {
    unsigned char* sc = malloc(SUBGRAPH_C_LEN);
    printf("running real stem7x7->bias_add->requantize->maxpool subgraph...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int src = mini_rpc_run_kernel(h, sa, SUBGRAPH_A_LEN, sa, 0, sc, SUBGRAPH_C_LEN);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("run_subgraph rc=%d wall_ms=%.3f\n", src, ms);
    if (src == 0) {
      FILE* out = fopen(SUBGRAPH_C_PATH, "wb");
      fwrite(sc, 1, SUBGRAPH_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against the ORT reference host-side\n", SUBGRAPH_C_PATH, SUBGRAPH_C_LEN);
    }
    free(sc);
  } else {
    printf("subgraph test data not found at %s, skipping\n", SUBGRAPH_A_PATH);
  }
  free(sa);

  /* Stage 3: ResNet-50 stage1/block1 -- conv10(1x1 reduce)->bias->requantize->conv17(3x3)->
   * bias->requantize->conv24(1x1 expand)->bias plus a conv30(1x1 downsample)->bias shortcut,
   * rescale-add-relu-clamp merge. Input is Stage 2's maxpool output (packed NCHWc), real
   * weights/scales from backbone.onnx. Only runs if the input was pushed first. */
  unsigned char* b1a = read_file_exact(BLOCK1_A_PATH, BLOCK1_A_LEN);
  if (b1a) {
    unsigned char* b1c = malloc(BLOCK1_C_LEN);
    printf("running real ResNet-50 stage1/block1 subgraph...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int b1rc = mini_rpc_run_kernel(h, b1a, BLOCK1_A_LEN, b1a, 0, b1c, BLOCK1_C_LEN);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("run_block1 rc=%d wall_ms=%.3f\n", b1rc, ms);
    if (b1rc == 0) {
      FILE* out = fopen(BLOCK1_C_PATH, "wb");
      fwrite(b1c, 1, BLOCK1_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against the ORT reference host-side\n", BLOCK1_C_PATH, BLOCK1_C_LEN);
    }
    free(b1c);
  } else {
    printf("block1 test data not found at %s, skipping\n", BLOCK1_A_PATH);
  }
  free(b1a);

  /* Real hex_boxhead_gemm_kernel.py-generated fp32 HVX GEMM for Mask R-CNN's box-head fc6 layer
   * (m=64,k=12544,n=512 -- k is the real fc6 reduction depth, m/n a real-shape-consistent slice
   * chosen so the weight buffer fits under the RPC transfer wall) -- only runs if
   * gen_boxhead_test_data.py's output was pushed first; skipped gracefully otherwise. Times the
   * on-device call for a real fp32-vs-int8 throughput comparison against this project's other
   * kernels. */
  unsigned char* ba = read_file_exact(BOXHEAD_A_PATH, BOXHEAD_A_LEN);
  unsigned char* bb = read_file_exact(BOXHEAD_B_PATH, BOXHEAD_B_LEN);
  if (ba && bb) {
    unsigned char* bc = malloc(BOXHEAD_C_LEN);
    printf("running real hex_boxhead_gemm kernel (m=64,k=12544,n=512)...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int brc = mini_rpc_run_kernel(h, ba, BOXHEAD_A_LEN, bb, BOXHEAD_B_LEN, bc, BOXHEAD_C_LEN);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double bms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("run_boxhead_gemm rc=%d wall_ms=%.3f\n", brc, bms);
    if (brc == 0) {
      FILE* out = fopen(BOXHEAD_C_PATH, "wb");
      fwrite(bc, 1, BOXHEAD_C_LEN, out);
      fclose(out);
      printf("wrote %s (%d bytes) -- verify against a numpy reference host-side\n", BOXHEAD_C_PATH, BOXHEAD_C_LEN);
    }
    free(bc);
  } else {
    printf("boxhead test data not found at %s / %s, skipping real-kernel test\n", BOXHEAD_A_PATH, BOXHEAD_B_PATH);
  }
  free(ba);
  free(bb);

  mini_rpc_close(h);
  printf("closed OK\n");
  return (result == 42 && rc == 0) ? 0 : 2;
}
