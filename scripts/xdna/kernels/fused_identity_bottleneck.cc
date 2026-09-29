// Chunked full-tensor kernels for a fused INT8 identity ResNet bottleneck.
#define NOCPP
#include <aie_api/aie.hpp>
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
#ifndef FUSED_C1_MMUL
#define FUSED_C1_MMUL 0
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
#if FUSED_C1_MMUL
  using MMUL1 = aie::mmul<8, 8, 8, int8, int8>;
  alignas(32) int8_t a10_tile[64], a11_tile[64];
  alignas(32) int8_t b10_tile[64], b11_tile[64];
  // Two spatial and two output-channel MMULs reuse each gathered activation
  // tile. We keep accumulation in int32 and apply bias/requantization once,
  // preserving the scalar reference's integer result exactly.
  for (int p0 = 0; p0 < pixels; p0 += 16) {
    const int valid0 = pixels - p0 < 8 ? pixels - p0 : 8;
    const int valid1 = pixels - (p0 + 8) < 8 ? pixels - (p0 + 8) : 8;
    for (int oc0 = 0; oc0 < outputs; oc0 += 16) {
      MMUL1 c100 = aie::zeros<acc32, 64>();
      MMUL1 c101 = aie::zeros<acc32, 64>();
      MMUL1 c110 = aie::zeros<acc32, 64>();
      MMUL1 c111 = aie::zeros<acc32, 64>();
      for (int ic0 = 0; ic0 < FUSED_C; ic0 += 8) {
        for (int m = 0; m < 16; ++m) for (int k = 0; k < 8; ++k) {
          const int p = p0 + m;
          const bool valid = (m < 8) ? (m < valid0) : ((m - 8) < valid1);
          const int8_t value = valid
              ? (int8_t)((int32_t)((const uint8_t *)input)[p * FUSED_C + ic0 + k] - 128)
              : 0;
          if (m < 8) a10_tile[m * 8 + k] = value;
          else a11_tile[(m - 8) * 8 + k] = value;
        }
        const int weight_offset = ic0 * outputs + oc0;
        for (int k = 0; k < 8; ++k) for (int n = 0; n < 8; ++n) {
          b10_tile[k * 8 + n] = weights[weight_offset + k * outputs + n];
          b11_tile[k * 8 + n] = weights[weight_offset + k * outputs + 8 + n];
        }
        auto av0 = aie::load_v<64>(a10_tile);
        auto av1 = aie::load_v<64>(a11_tile);
        auto bv0 = aie::load_v<64>(b10_tile);
        auto bv1 = aie::load_v<64>(b11_tile);
        c100.mac(av0, bv0); c101.mac(av0, bv1);
        c110.mac(av1, bv0); c111.mac(av1, bv1);
      }
      auto s100 = c100.to_vector<int32>(); auto s101 = c101.to_vector<int32>();
      auto s110 = c110.to_vector<int32>(); auto s111 = c111.to_vector<int32>();
      for (int m = 0; m < valid0; ++m) for (int n = 0; n < 16; ++n) {
        const int32_t dot = n < 8 ? s100[m * 8 + n] : s101[m * 8 + n - 8];
        int32_t q = 128 + round_shift_even(dot + bias[oc0 + n], FUSED_SHIFT1);
        if (q < 128) q = 128; else if (q > 255) q = 255;
        bundle[(p0 + m) * FUSED_MID + chunk * outputs + oc0 + n] = (uint8_t)q;
      }
      for (int m = 0; m < valid1; ++m) for (int n = 0; n < 16; ++n) {
        const int32_t dot = n < 8 ? s110[m * 8 + n] : s111[m * 8 + n - 8];
        int32_t q = 128 + round_shift_even(dot + bias[oc0 + n], FUSED_SHIFT1);
        if (q < 128) q = 128; else if (q > 255) q = 255;
        bundle[(p0 + 8 + m) * FUSED_MID + chunk * outputs + oc0 + n] = (uint8_t)q;
      }
    }
  }
#else
  for (int p = 0; p < pixels; ++p) for (int oc = 0; oc < outputs; ++oc) {
    int32_t acc = bias[oc];
    for (int ic = 0; ic < FUSED_C; ++ic)
      acc += ((int32_t)((uint8_t)input[p * FUSED_C + ic] - 128)) * (int32_t)weights[oc * FUSED_C + ic];
    int32_t q = 128 + round_shift_even(acc, FUSED_SHIFT1);
    if (q < 128) q = 128; else if (q > 255) q = 255;
    bundle[p * FUSED_MID + chunk * outputs + oc] = (uint8_t)q;
  }
#endif
}

extern "C" void fused_bottleneck_conv2_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk, int32_t channel_offset) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS2_OFFSET);
  const int outputs = (FUSED_MID / 2) / FUSED_C2_CHUNKS;
  const int pixels = FUSED_OUT_W * FUSED_OUT_H;
  const uint8_t *q1 = bundle;
  using MMUL = aie::mmul<8, 8, 8, int8, int8>;
  alignas(32) int8_t a0_tile[64], a1_tile[64];
  alignas(32) int8_t b0_tile[64], b1_tile[64];
  // The 2x2 MMUL schedule raises utilization when both the pixel and output
  // tiles are large enough. Small late-stage tensors use the scalar path below
  // to avoid spending more time gathering padded tiles than doing MACs.
  if (outputs >= 16 && (outputs % 16) == 0 && pixels >= 16) {
  for (int p0 = 0; p0 < pixels; p0 += 16) {
    const int valid0 = pixels - p0 < 8 ? pixels - p0 : 8;
    const int valid1 = pixels - (p0 + 8) < 8 ? pixels - (p0 + 8) : 8;
    for (int oc0 = 0; oc0 < outputs; oc0 += 16) {
      MMUL c00 = aie::zeros<acc32, 64>();
      MMUL c01 = aie::zeros<acc32, 64>();
      MMUL c10 = aie::zeros<acc32, 64>();
      MMUL c11 = aie::zeros<acc32, 64>();
      for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
        for (int ic0 = 0; ic0 < FUSED_MID; ic0 += 8) {
          for (int m = 0; m < 16; ++m) {
            const int p = p0 + m;
            const int y = p / FUSED_OUT_W;
            const int x = p - y * FUSED_OUT_W;
            const int iy = y * FUSED_CONV2_STRIDE + ky - 1;
            const int ix = x * FUSED_CONV2_STRIDE + kx - 1;
            const bool valid = (m < 8) ? (m < valid0) : ((m - 8) < valid1);
            for (int k = 0; k < 8; ++k) {
              int8_t value = 0;
              if (valid && iy >= 0 && iy < FUSED_H && ix >= 0 && ix < FUSED_W) {
                const int ip = iy * FUSED_W + ix;
                value = (int8_t)((int32_t)q1[ip * FUSED_MID + ic0 + k] - 128);
              }
              if (m < 8) a0_tile[m * 8 + k] = value;
              else a1_tile[(m - 8) * 8 + k] = value;
            }
          }
          const int weight_offset = ((ky * 3 + kx) * FUSED_MID + ic0) * outputs + oc0;
          for (int k = 0; k < 8; ++k) for (int n = 0; n < 8; ++n) {
            b0_tile[k * 8 + n] = weights[weight_offset + k * outputs + n];
            b1_tile[k * 8 + n] = weights[weight_offset + k * outputs + 8 + n];
          }
          auto av0 = aie::load_v<64>(a0_tile);
          auto av1 = aie::load_v<64>(a1_tile);
          auto bv0 = aie::load_v<64>(b0_tile);
          auto bv1 = aie::load_v<64>(b1_tile);
          c00.mac(av0, bv0);
          c01.mac(av0, bv1);
          c10.mac(av1, bv0);
          c11.mac(av1, bv1);
        }
      }
      auto s00 = c00.to_vector<int32>();
      auto s01 = c01.to_vector<int32>();
      auto s10 = c10.to_vector<int32>();
      auto s11 = c11.to_vector<int32>();
      for (int m = 0; m < valid0; ++m) for (int n = 0; n < 8; ++n) {
        int32_t q0 = 128 + round_shift_even(s00[m * 8 + n] + bias[oc0 + n], FUSED_SHIFT2);
        int32_t q1v = 128 + round_shift_even(s01[m * 8 + n] + bias[oc0 + 8 + n], FUSED_SHIFT2);
        if (q0 < 128) q0 = 128; else if (q0 > 255) q0 = 255;
        if (q1v < 128) q1v = 128; else if (q1v > 255) q1v = 255;
        output[(p0 + m) * (FUSED_MID / 2) + chunk * outputs + oc0 + n] = (uint8_t)q0;
        output[(p0 + m) * (FUSED_MID / 2) + chunk * outputs + oc0 + 8 + n] = (uint8_t)q1v;
      }
      for (int m = 0; m < valid1; ++m) for (int n = 0; n < 8; ++n) {
        int32_t q0 = 128 + round_shift_even(s10[m * 8 + n] + bias[oc0 + n], FUSED_SHIFT2);
        int32_t q1v = 128 + round_shift_even(s11[m * 8 + n] + bias[oc0 + 8 + n], FUSED_SHIFT2);
        if (q0 < 128) q0 = 128; else if (q0 > 255) q0 = 255;
        if (q1v < 128) q1v = 128; else if (q1v > 255) q1v = 255;
        output[(p0 + 8 + m) * (FUSED_MID / 2) + chunk * outputs + oc0 + n] = (uint8_t)q0;
        output[(p0 + 8 + m) * (FUSED_MID / 2) + chunk * outputs + oc0 + 8 + n] = (uint8_t)q1v;
      }
    }
  }
  } else {
    for (int y = 0; y < FUSED_OUT_H; ++y) for (int x = 0; x < FUSED_OUT_W; ++x) {
      int p = y * FUSED_OUT_W + x;
      for (int oc = 0; oc < outputs; ++oc) {
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
  }
  (void)channel_offset;
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
  using MMUL3 = aie::mmul<8,  8, 8, int8, int8>;
  alignas(32) int8_t a30_tile[64], a31_tile[64];
  alignas(32) int8_t b30_tile[64], b31_tile[64];
  if (outputs >= 16 && (outputs % 16) == 0 && pixels >= 16) {
    for (int p0 = 0; p0 < pixels; p0 += 16) {
      const int valid0 = pixels - p0 < 8 ? pixels - p0 : 8;
      const int valid1 = pixels - (p0 + 8) < 8 ? pixels - (p0 + 8) : 8;
      for (int oc0 = 0; oc0 < outputs; oc0 += 16) {
        MMUL3 c300 = aie::zeros<acc32, 64>();
        MMUL3 c301 = aie::zeros<acc32, 64>();
        MMUL3 c310 = aie::zeros<acc32, 64>();
        MMUL3 c311 = aie::zeros<acc32, 64>();
        for (int ic0 = 0; ic0 < FUSED_MID; ic0 += 8) {
          for (int m = 0; m < 16; ++m) for (int k = 0; k < 8; ++k) {
            const int p = p0 + m;
            const bool valid = (m < 8) ? (m < valid0) : ((m - 8) < valid1);
            const uint8_t *source = ic0 + k < half ? q2a : q2b;
            const int source_k = ic0 + k < half ? ic0 + k : ic0 + k - half;
            const int8_t value = valid ? (int8_t)((int32_t)source[p * half + source_k] - 128) : 0;
            if (m < 8) a30_tile[m * 8 + k] = value;
            else a31_tile[(m - 8) * 8 + k] = value;
          }
          for (int k = 0; k < 8; ++k) for (int n = 0; n < 8; ++n) {
            b30_tile[k * 8 + n] = weights[(ic0 + k) * outputs + oc0 + n];
            b31_tile[k * 8 + n] = weights[(ic0 + k) * outputs + oc0 + 8 + n];
          }
          auto av0 = aie::load_v<64>(a30_tile);
          auto av1 = aie::load_v<64>(a31_tile);
          auto bv0 = aie::load_v<64>(b30_tile);
          auto bv1 = aie::load_v<64>(b31_tile);
          c300.mac(av0, bv0); c301.mac(av0, bv1);
          c310.mac(av1, bv0); c311.mac(av1, bv1);
        }
        auto s300 = c300.to_vector<int32>(); auto s301 = c301.to_vector<int32>();
        auto s310 = c310.to_vector<int32>(); auto s311 = c311.to_vector<int32>();
        for (int m = 0; m < valid0; ++m) for (int n = 0; n < 16; ++n) {
          const int p = p0 + m;
          const int oc = oc0 + n;
          const int32_t acc = (n < 8 ? s300[m * 8 + n] : s301[m * 8 + n - 8]) + bias[oc];
          int32_t q3 = 128 + round_shift_even(acc, FUSED_SHIFT3);
          if (q3 < 0) q3 = 0; else if (q3 > 255) q3 = 255;
#if FUSED_PROJECTION
          const int common = FUSED_MAIN_RESIDUAL_SHIFT < FUSED_SKIP_RESIDUAL_SHIFT
              ? (FUSED_MAIN_RESIDUAL_SHIFT < 0 ? -FUSED_MAIN_RESIDUAL_SHIFT : 0)
              : (FUSED_SKIP_RESIDUAL_SHIFT < 0 ? -FUSED_SKIP_RESIDUAL_SHIFT : 0);
          int64_t sum = scale_pow2(q3 - 128, FUSED_MAIN_RESIDUAL_SHIFT, common) +
                        scale_pow2((int32_t)skip[p * FUSED_OUT_C + chunk * outputs + oc] - 128, FUSED_SKIP_RESIDUAL_SHIFT, common);
#else
          const int common = FUSED_RESIDUAL_SHIFT < FUSED_INPUT_SHIFT
              ? (FUSED_RESIDUAL_SHIFT < 0 ? -FUSED_RESIDUAL_SHIFT : 0)
              : (FUSED_INPUT_SHIFT < 0 ? -FUSED_INPUT_SHIFT : 0);
          int64_t sum = scale_pow2(q3 - 128, FUSED_RESIDUAL_SHIFT, common) +
                        scale_pow2((int8_t)residual[p * FUSED_C + chunk * outputs + oc], FUSED_INPUT_SHIFT, common);
#endif
          int64_t value = round_shift_even64(sum, common);
          if (value < 0) value = 0; else if (value > 127) value = 127;
          output[p * FUSED_OUT_C + chunk * outputs + oc] = (uint8_t)(128 + value);
        }
        for (int m = 0; m < valid1; ++m) for (int n = 0; n < 16; ++n) {
          const int p = p0 + 8 + m, oc = oc0 + n;
          const int32_t acc = (n < 8 ? s310[m * 8 + n] : s311[m * 8 + n - 8]) + bias[oc];
          int32_t q3 = 128 + round_shift_even(acc, FUSED_SHIFT3);
          if (q3 < 0) q3 = 0; else if (q3 > 255) q3 = 255;
#if FUSED_PROJECTION
          const int common = FUSED_MAIN_RESIDUAL_SHIFT < FUSED_SKIP_RESIDUAL_SHIFT
              ? (FUSED_MAIN_RESIDUAL_SHIFT < 0 ? -FUSED_MAIN_RESIDUAL_SHIFT : 0)
              : (FUSED_SKIP_RESIDUAL_SHIFT < 0 ? -FUSED_SKIP_RESIDUAL_SHIFT : 0);
          int64_t sum = scale_pow2(q3 - 128, FUSED_MAIN_RESIDUAL_SHIFT, common) +
                        scale_pow2((int32_t)skip[p * FUSED_OUT_C + chunk * outputs + oc] - 128, FUSED_SKIP_RESIDUAL_SHIFT, common);
#else
          const int common = FUSED_RESIDUAL_SHIFT < FUSED_INPUT_SHIFT
              ? (FUSED_RESIDUAL_SHIFT < 0 ? -FUSED_RESIDUAL_SHIFT : 0)
              : (FUSED_INPUT_SHIFT < 0 ? -FUSED_INPUT_SHIFT : 0);
          int64_t sum = scale_pow2(q3 - 128, FUSED_RESIDUAL_SHIFT, common) +
                        scale_pow2((int8_t)residual[p * FUSED_C + chunk * outputs + oc], FUSED_INPUT_SHIFT, common);
#endif
          int64_t value = round_shift_even64(sum, common);
          if (value < 0) value = 0; else if (value > 127) value = 127;
          output[p * FUSED_OUT_C + chunk * outputs + oc] = (uint8_t)(128 + value);
        }
      }
    }
    return;
  }
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
