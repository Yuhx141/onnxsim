// Chunked full-tensor kernels for a fused INT8 identity ResNet bottleneck.
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
#ifndef FUSED_MID
#error "FUSED_MID must be defined"
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
#ifndef FUSED_CONV2_STRIDE
#define FUSED_CONV2_STRIDE 1
#endif
#ifndef FUSED_SKIP_STRIDE
#define FUSED_SKIP_STRIDE 1
#endif
#ifndef FUSED_PROJECTION
#define FUSED_PROJECTION 0
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
static int64_t round_shift_even64(int64_t value, int shift) {
  if (shift <= 0) return value;
  uint64_t mag = value < 0 ? (uint64_t)(-value) : (uint64_t)value;
  uint64_t q = mag >> shift, rem = mag & ((((uint64_t)1) << shift) - 1);
  uint64_t half = ((uint64_t)1) << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) ++q;
  return value < 0 ? -(int64_t)q : (int64_t)q;
}
static int64_t scale_pow2(int32_t value, int exponent, int common_shift) {
  return ((int64_t)value) * (((int64_t)1) << (exponent + common_shift));
}

extern "C" void fused_bottleneck_conv1_chunk(const int8_t *input, const uint8_t *params, uint8_t *bundle, int32_t chunk) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS1_OFFSET);
  const int outputs = FUSED_MID / FUSED_C1_CHUNKS;
  const int pixels = FUSED_W * FUSED_H;
  for (int p = 0; p < pixels; ++p) for (int oc = 0; oc < outputs; ++oc) {
    int32_t acc = bias[oc];
    for (int ic = 0; ic < FUSED_C; ++ic)
      acc += ((int32_t)((uint8_t)input[p * FUSED_C + ic] - 128)) * (int32_t)weights[oc * FUSED_C + ic];
    int32_t q = 128 + round_shift_even(acc, FUSED_SHIFT1);
    if (q < 128) q = 128; else if (q > 255) q = 255;
    bundle[p * FUSED_MID + chunk * outputs + oc] = (uint8_t)q;
  }
}

extern "C" void fused_bottleneck_conv2_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk, int32_t channel_offset) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS2_OFFSET);
  const int outputs = (FUSED_MID / 2) / FUSED_C2_CHUNKS;
  const int pixels = FUSED_OUT_W * FUSED_OUT_H;
  const uint8_t *q1 = bundle;
  for (int y = 0; y < FUSED_OUT_H; ++y) for (int x = 0; x < FUSED_OUT_W; ++x) {
    int p = y * FUSED_OUT_W + x;
    for (int oc = 0; oc < outputs; ++oc) {
      int full_oc = channel_offset + chunk * outputs + oc;
      int32_t acc = bias[oc];
      for (int ky = 0; ky < 3; ++ky) {
        int iy = y * FUSED_CONV2_STRIDE + ky - 1;
        if (iy < 0 || iy >= FUSED_H) continue;
        for (int kx = 0; kx < 3; ++kx) {
          int ix = x * FUSED_CONV2_STRIDE + kx - 1;
          if (ix < 0 || ix >= FUSED_W) continue;
          int ip = iy * FUSED_W + ix;
          for (int ic = 0; ic < FUSED_MID; ++ic) {
            int wi = ((oc * FUSED_MID + ic) * 3 + ky) * 3 + kx;
            acc += ((int32_t)q1[ip * FUSED_MID + ic] - 128) * (int32_t)weights[wi];
          }
        }
      }
      int32_t q = 128 + round_shift_even(acc, FUSED_SHIFT2);
      if (q < 128) q = 128; else if (q > 255) q = 255;
      output[p * (FUSED_MID / 2) + chunk * outputs + oc] = (uint8_t)q;
    }
  }
  (void)pixels;
}

extern "C" void fused_bottleneck_copy_residual(const uint8_t *stage1, uint8_t *stage2a) {
  const int pixels = FUSED_W * FUSED_H;
  const uint8_t *residual = stage1 + pixels * FUSED_MID;
  uint8_t *destination = stage2a + pixels * (FUSED_MID / 2);
  for (int i = 0; i < pixels * FUSED_C; ++i) destination[i] = residual[i];
}

extern "C" void fused_bottleneck_conv2_chunk_b(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk, int32_t channel_offset) {
  fused_bottleneck_conv2_chunk(bundle, params, output, chunk, channel_offset);
}

extern "C" void fused_bottleneck_conv3_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS3_OFFSET);
  const int outputs = FUSED_OUT_C / FUSED_C3_CHUNKS;
  const int pixels = FUSED_OUT_W * FUSED_OUT_H;
  const int half = FUSED_MID / 2;
  const uint8_t *q2a = bundle;
#if FUSED_PROJECTION
  const uint8_t *q2b = bundle + pixels * half;
  const uint8_t *skip = bundle + pixels * FUSED_MID;
#else
  const int8_t *residual = (const int8_t *)(bundle + pixels * FUSED_MID);
  const uint8_t *q2b = bundle + pixels * half;
#endif
  for (int p = 0; p < pixels; ++p) for (int oc = 0; oc < outputs; ++oc) {
    int out_channel = chunk * outputs + oc;
    int32_t acc = bias[oc];
    for (int ic = 0; ic < half; ++ic) {
      acc += ((int32_t)q2a[p * half + ic] - 128) * (int32_t)weights[oc * FUSED_MID + ic];
      acc += ((int32_t)q2b[p * half + ic] - 128) * (int32_t)weights[oc * FUSED_MID + half + ic];
    }
    int32_t q3 = 128 + round_shift_even(acc, FUSED_SHIFT3);
    if (q3 < 0) q3 = 0; else if (q3 > 255) q3 = 255;
#if FUSED_PROJECTION
    const int common = FUSED_MAIN_RESIDUAL_SHIFT < FUSED_SKIP_RESIDUAL_SHIFT
        ? (FUSED_MAIN_RESIDUAL_SHIFT < 0 ? -FUSED_MAIN_RESIDUAL_SHIFT : 0)
        : (FUSED_SKIP_RESIDUAL_SHIFT < 0 ? -FUSED_SKIP_RESIDUAL_SHIFT : 0);
    int64_t sum = scale_pow2(q3 - 128, FUSED_MAIN_RESIDUAL_SHIFT, common) +
                  scale_pow2((int32_t)skip[p * FUSED_OUT_C + out_channel] - 128,
                             FUSED_SKIP_RESIDUAL_SHIFT, common);
#else
    const int common = FUSED_RESIDUAL_SHIFT < FUSED_INPUT_SHIFT
        ? (FUSED_RESIDUAL_SHIFT < 0 ? -FUSED_RESIDUAL_SHIFT : 0)
        : (FUSED_INPUT_SHIFT < 0 ? -FUSED_INPUT_SHIFT : 0);
    int64_t sum = scale_pow2(q3 - 128, FUSED_RESIDUAL_SHIFT, common) +
                  scale_pow2((int8_t)residual[p * FUSED_C + out_channel], FUSED_INPUT_SHIFT, common);
#endif
    int64_t value = round_shift_even64(sum, common);
    if (value < 0) value = 0; else if (value > 127) value = 127;
    output[p * FUSED_OUT_C + out_channel] = (uint8_t)(128 + value);
  }
}
