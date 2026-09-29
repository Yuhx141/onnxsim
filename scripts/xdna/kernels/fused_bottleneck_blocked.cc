// Vectorized ResNet bottleneck kernels for the linked stage design.
//
// Activations between blocks (and at the stage boundary) are uint8-with-zero-
// point-128 bytes in an 8-channel *blocked* layout [C/8][pixel][8]; the shim
// DMA converts to/from host NHWC. Inside a block, conv1/conv2 outputs are
// signed int8 (already offset-removed). Weights are pre-tiled by
// blocked_stage.pack_blocked_params so every MMUL operand is one 64-byte load.
// Results are bit-identical to fused_identity_bottleneck.cc: int32
// accumulation, round-half-even shift (aie conv_even srs), int8 saturation.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef FUSED_W
#define FUSED_W 8
#endif
#ifndef FUSED_H
#define FUSED_H 8
#endif
#ifndef FUSED_C
#define FUSED_C 64
#endif
#ifndef FUSED_MID
#define FUSED_MID 64
#endif
#ifndef FUSED_OUT_C
#define FUSED_OUT_C 256
#endif

namespace {
constexpr int W = FUSED_W, H = FUSED_H, P = W * H;
constexpr int TILES = P / 8;
constexpr int PW = W + 2, PADP = (H + 2) * PW;
constexpr int ICB1 = FUSED_C / 8, OCB1 = FUSED_MID / 8;
constexpr int ICB2 = FUSED_MID / 8, OCB2 = (FUSED_MID / 2) / 8;
constexpr int ICB3 = FUSED_MID / 8, OCB3 = FUSED_OUT_C / 8;
constexpr int OCBS = FUSED_OUT_C / 8;

constexpr int pos(int v) { return v > 0 ? v : 0; }

using MMUL = aie::mmul<8, 8, 8, int8, int8>;
using v64 = aie::vector<int8, 64>;

inline v64 flip(v64 v) { return aie::bit_xor(v, aie::broadcast<int8, 64>((int8_t)-128)); }

inline aie::vector<int32, 64> bias_tile(const int32_t *b) {
  aie::vector<int32, 8> v = aie::load_unaligned_v<8>(b);
  aie::vector<int32, 16> v2 = aie::concat(v, v);
  aie::vector<int32, 32> v4 = aie::concat(v2, v2);
  return aie::concat(v4, v4);
}

inline v64 load_tile(const int8_t *p) { return aie::load_unaligned_v<64>(p); }
}  // namespace

#ifdef BLK_CONV1
extern "C" void fused_bottleneck_conv1_chunk(const int8_t *input, const uint8_t *params, uint8_t *bundle, int32_t chunk) {
  (void)chunk;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS1_OFFSET);
  int8_t *out = (int8_t *)bundle;
  const v64 zero = aie::zeros<int8, 64>();
  for (int i = 0; i < OCB1 * PADP * 8; i += 64) aie::store_unaligned_v(out + i, zero);
  for (int oc = 0; oc < OCB1; ++oc) {
    for (int t = 0; t < TILES; ++t) {
      MMUL c(bias_tile(bias + oc * 8));
      for (int icb = 0; icb < ICB1; ++icb) {
        v64 a = flip(load_tile(input + (icb * P + t * 8) * 8));
        v64 b = load_tile(weights + (oc * ICB1 + icb) * 64);
        c.mac(a, b);
      }
      v64 v = aie::max(c.to_vector<int8>(pos(FUSED_SHIFT1)), zero);
      const int y = (t * 8) / W, x0 = (t * 8) % W;
      aie::store_unaligned_v(out + (oc * PADP + (y + 1) * PW + x0 + 1) * 8, v);
    }
  }
}

#endif  // BLK_CONV1

#if defined(BLK_CONV2A) || defined(BLK_CONV2B)
static void conv2_impl(const uint8_t *bundle, const uint8_t *params, uint8_t *output) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  const int8_t *in = (const int8_t *)bundle;
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS2_OFFSET);
  int8_t *out = (int8_t *)output;
  const v64 zero = aie::zeros<int8, 64>();
  for (int oc = 0; oc < OCB2; ++oc) {
    for (int t = 0; t < TILES; ++t) {
      const int y = (t * 8) / W, x0 = (t * 8) % W;
      MMUL c(bias_tile(bias + oc * 8));
      for (int tap = 0; tap < 9; ++tap) {
        const int ky = tap / 3, kx = tap % 3;
        for (int icb = 0; icb < ICB2; ++icb) {
          v64 a = load_tile(in + (icb * PADP + (y + ky) * PW + x0 + kx) * 8);
          v64 b = load_tile(weights + ((oc * 9 + tap) * ICB2 + icb) * 64);
          c.mac(a, b);
        }
      }
      v64 v = aie::max(c.to_vector<int8>(pos(FUSED_SHIFT2)), zero);
#if defined(FUSED_DBG) && FUSED_DBG == 4
      v = load_tile(in + (oc * PADP + (y + 1) * PW + x0 + 1) * 8);
#endif
      aie::store_unaligned_v(out + (oc * P + t * 8) * 8, v);
    }
  }
}

#endif

#ifdef BLK_CONV2A
extern "C" void fused_bottleneck_conv2_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t, int32_t) {
  conv2_impl(bundle, params, output);
}
#endif
#ifdef BLK_CONV2B
extern "C" void fused_bottleneck_conv2_chunk_b(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t, int32_t) {
  conv2_impl(bundle, params, output);
}
#endif

#ifdef BLK_SKIP
extern "C" void fused_bottleneck_skip_chunk(const int8_t *input, const uint8_t *params, uint8_t *output, int32_t chunk) {
  (void)chunk;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_SKIP_BIAS_OFFSET);
  int8_t *out = (int8_t *)output;
  for (int oc = 0; oc < OCBS; ++oc) {
    for (int t = 0; t < TILES; ++t) {
      MMUL c(bias_tile(bias + oc * 8));
      for (int icb = 0; icb < ICB1; ++icb) {
        v64 a = flip(load_tile(input + (icb * P + t * 8) * 8));
        v64 b = load_tile(weights + (oc * ICB1 + icb) * 64);
        c.mac(a, b);
      }
      aie::store_unaligned_v(out + (oc * P + t * 8) * 8, c.to_vector<int8>(pos(FUSED_SKIP_SHIFT)));
    }
  }
}

#endif

#ifdef BLK_IDENTITY
extern "C" void fused_bottleneck_identity_skip(const int8_t *input, int8_t *output) {
  for (int i = 0; i < P * FUSED_C; i += 64) aie::store_unaligned_v(output + i, flip(load_tile(input + i)));
}
#endif

#ifdef BLK_CONV3
extern "C" void fused_bottleneck_conv3_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk) {
  (void)chunk;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  const int8_t *in = (const int8_t *)bundle;
  const int8_t *skip = in + P * FUSED_MID;
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS3_OFFSET);
  int8_t *out = (int8_t *)output;
  constexpr int EA = FUSED_MAIN_RESIDUAL_SHIFT, EB = FUSED_SKIP_RESIDUAL_SHIFT;
  constexpr int common = (EA < EB) ? pos(-EA) : pos(-EB);
  const v64 zero = aie::zeros<int8, 64>();
  for (int oc = 0; oc < OCB3; ++oc) {
    for (int t = 0; t < TILES; ++t) {
      MMUL c(bias_tile(bias + oc * 8));
      for (int icb = 0; icb < ICB3; ++icb) {
        v64 a = load_tile(in + (icb * P + t * 8) * 8);
        v64 b = load_tile(weights + (oc * ICB3 + icb) * 64);
        c.mac(a, b);
      }
      v64 q3 = c.to_vector<int8>(pos(FUSED_SHIFT3));
      v64 r = load_tile(skip + (oc * P + t * 8) * 8);
      aie::accum<acc32, 64> a1, a2;
      a1.from_vector(q3, EA + common);
      a2.from_vector(r, EB + common);
      aie::accum<acc32, 64> sum = aie::add(a1, a2);
      v64 v = aie::max(sum.to_vector<int8>(common), zero);
#if defined(FUSED_DBG) && FUSED_DBG == 3
      v = oc < ICB3 ? load_tile(in + (oc * P + t * 8) * 8) : zero;
#elif defined(FUSED_DBG) && FUSED_DBG == 1
      v = r;
#elif defined(FUSED_DBG) && FUSED_DBG == 2
      v = q3;
#endif
      aie::store_unaligned_v(out + (oc * P + t * 8) * 8, flip(v));
    }
  }
}
#endif
