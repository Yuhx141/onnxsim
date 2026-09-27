#include <stdint.h>

#ifndef FUSED_W
#error "FUSED_W must be defined"
#endif
#ifndef FUSED_H
#error "FUSED_H must be defined"
#endif
#ifndef FUSED_C
#error "FUSED_C must be defined"
#endif
#ifndef FUSED_OUT_H
#define FUSED_OUT_H FUSED_H
#endif
#ifndef FUSED_OUT_W
#define FUSED_OUT_W FUSED_W
#endif
#ifndef FUSED_OUT_C
#define FUSED_OUT_C FUSED_C
#endif
#ifndef FUSED_SKIP_STRIDE
#define FUSED_SKIP_STRIDE 1
#endif
#ifndef FUSED_SKIP_CHUNKS
#define FUSED_SKIP_CHUNKS 1
#endif

static int32_t round_shift_even(int32_t value, int shift) {
  if (shift <= 0) return value;
  int64_t wide = value;
  uint64_t mag = wide < 0 ? (uint64_t)(-wide) : (uint64_t)wide;
  uint64_t q = mag >> shift, rem = mag & ((((uint64_t)1) << shift) - 1);
  uint64_t half = ((uint64_t)1) << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) ++q;
  return wide < 0 ? -(int32_t)q : (int32_t)q;
}

// Projection emits quantized uint8 bytes.
extern "C" void fused_bottleneck_skip_chunk(const int8_t *input,
                                             const uint8_t *params,
                                             uint8_t *output,
                                             int32_t chunk) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_SKIP_BIAS_OFFSET);
  const int outputs = FUSED_OUT_C / FUSED_SKIP_CHUNKS;
  for (int y = 0; y < FUSED_OUT_H; ++y) for (int x = 0; x < FUSED_OUT_W; ++x) {
    const int p = y * FUSED_OUT_W + x;
    const int ip = (y * FUSED_SKIP_STRIDE) * FUSED_W + x * FUSED_SKIP_STRIDE;
    for (int oc = 0; oc < outputs; ++oc) {
      int32_t acc = bias[oc];
      for (int ic = 0; ic < FUSED_C; ++ic)
        acc += ((int32_t)((uint8_t)input[ip * FUSED_C + ic] - 128)) * (int32_t)weights[oc * FUSED_C + ic];
      int32_t q = 128 + round_shift_even(acc, FUSED_SKIP_SHIFT);
      if (q < 0) q = 0; else if (q > 255) q = 255;
      output[p * FUSED_OUT_C + chunk * outputs + oc] = (uint8_t)q;
    }
  }
}
