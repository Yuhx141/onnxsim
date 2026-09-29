// Vectorized ResNet bottleneck kernels for the linked stage design.
//
// Activations between blocks (and at the stage boundary) are uint8-with-zero-
// point-128 bytes in an 8-channel *blocked* layout [C/8][pixel][8]; the shim
// DMA converts to/from host NHWC. Inside a block, conv1/conv2 outputs are
// signed int8 (already offset-removed). Weights are pre-tiled by
// blocked_stage.pack_blocked_params into 8x8 MMUL B tiles ordered
// [out-block][k-tile], so every MMUL operand is one 64-byte load. Results are
// bit-identical to fused_identity_bottleneck.cc: int32 accumulation,
// round-half-even shift (aie conv_even srs), int8 saturation.
//
// Generality: any H/W (tiles are 8 flattened output pixels; tiles that cross
// image rows or run past the pixel count use small scalar gathers/scatters),
// conv2 stride 1 or 2, projection skips with stride, and weight streams split
// into several output-channel chunks (`chunk` selects the output-block range).
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
#ifndef FUSED_OUT_W
#define FUSED_OUT_W FUSED_W
#endif
#ifndef FUSED_OUT_H
#define FUSED_OUT_H FUSED_H
#endif
#ifndef FUSED_CONV2_STRIDE
#define FUSED_CONV2_STRIDE 1
#endif
#ifndef FUSED_SKIP_STRIDE
#define FUSED_SKIP_STRIDE FUSED_CONV2_STRIDE
#endif
#ifndef FUSED_C1_CHUNKS
#define FUSED_C1_CHUNKS 1
#endif
#ifndef FUSED_C2_CHUNKS
#define FUSED_C2_CHUNKS 1
#endif
#ifndef FUSED_C3_CHUNKS
#define FUSED_C3_CHUNKS 1
#endif
#ifndef FUSED_SKIP_CHUNKS
#define FUSED_SKIP_CHUNKS 1
#endif

// FUSED_RT_SHIFTS: the requantization shifts are runtime values read from a 6 x int32
// header stored right after each weight chunk's payload (at FUSED_HDR_OFFSET), so one
// compiled block can serve several same-shaped blocks with different scales.
#ifdef FUSED_RT_SHIFTS
#define RT_HDR(i) (((const int32_t *)(params + FUSED_HDR_OFFSET))[i])
#define SHIFT1_V pos(RT_HDR(0))
#define SHIFT2_V pos(RT_HDR(1))
#define SHIFT3_V pos(RT_HDR(2))
#define SKIPSHIFT_V pos(RT_HDR(3))
#define EA_V (RT_HDR(4))
#define EB_V (RT_HDR(5))
#else
#define SHIFT1_V pos(FUSED_SHIFT1)
#define SHIFT2_V pos(FUSED_SHIFT2)
#define SHIFT3_V pos(FUSED_SHIFT3)
#define SKIPSHIFT_V pos(FUSED_SKIP_SHIFT)
#define EA_V (FUSED_MAIN_RESIDUAL_SHIFT)
#define EB_V (FUSED_SKIP_RESIDUAL_SHIFT)
#endif

namespace {
constexpr int W = FUSED_W, H = FUSED_H, P = W * H;
constexpr int OW = FUSED_OUT_W, OH = FUSED_OUT_H, OP = OW * OH;
constexpr int S = FUSED_CONV2_STRIDE, SS = FUSED_SKIP_STRIDE;
constexpr int PW = W + 2, PADP = (H + 2) * PW;
constexpr int CB = FUSED_C / 8, MB = FUSED_MID / 8, HB = MB / 2, OB = FUSED_OUT_C / 8;
constexpr int T1 = (P + 7) / 8, TO = (OP + 7) / 8;
constexpr int NB1 = MB / FUSED_C1_CHUNKS;
constexpr int NB2 = HB / FUSED_C2_CHUNKS;
constexpr int NB3 = OB / FUSED_C3_CHUNKS;
constexpr int NBS = OB / FUSED_SKIP_CHUNKS;

constexpr int pos(int v) { return v > 0 ? v : 0; }
constexpr int pick_group(int nb) { return nb % 4 == 0 ? 4 : (nb % 2 == 0 ? 2 : 1); }
// 3x3 taps that touch at least one real input pixel; the packed weights hold only these.
constexpr bool tap_valid(int tap) {
  const int ky = tap / 3, kx = tap % 3;
  for (int oy = 0; oy < OH; ++oy)
    for (int ox = 0; ox < OW; ++ox) {
      const int iy = oy * S + ky - 1, ix = ox * S + kx - 1;
      if (iy >= 0 && iy < H && ix >= 0 && ix < W) return true;
    }
  return false;
}
constexpr int count_taps() { int n = 0; for (int t = 0; t < 9; ++t) n += tap_valid(t) ? 1 : 0; return n; }
constexpr int NTAPS = count_taps();
struct TapTable { int8_t tap[9]; };
constexpr TapTable make_tap_table() {
  TapTable table{};
  int n = 0;
  for (int t = 0; t < 9; ++t) if (tap_valid(t)) table.tap[n++] = (int8_t)t;
  return table;
}
constexpr TapTable TAPS = make_tap_table();  // evaluated at compile time; runtime is one lookup

constexpr bool ROW_TILES1 = (W % 8 == 0);  // conv1 output tile = 8 pixels of one row
constexpr bool ROW_TILES2 = (S == 1 && W % 8 == 0 && OW == W);

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

inline void copy8(int8_t *dst, const int8_t *src) { __builtin_memcpy(dst, src, 8); }

// Store `rows` (<= 8) valid pixel rows of tile `v` at dst (pixel-major, 8 bytes each).
inline void store_rows(int8_t *dst, v64 v, int rows) {
  if (rows >= 8) {
    aie::store_unaligned_v(dst, v);
  } else {
    alignas(64) int8_t scratch[64];
    aie::store_v(scratch, v);
    for (int r = 0; r < rows; ++r) copy8(dst + r * 8, scratch + r * 8);
  }
}

// Generic tiled GEMM over one weight chunk: `nb` output blocks, `T` pixel tiles,
// `KT` k-tiles. a_get(t, kk) -> A tile; epi(ocl, t, acc) consumes each result.
template <int G, typename AGet, typename Epi>
inline void tiled_gemm(int nb, int T, int KT, const int8_t *w, const int32_t *bias, AGet a_get, Epi epi) {
  for (int t = 0; t < T; ++t) {
    for (int og = 0; og < nb; og += G) {
      MMUL c[G];
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < G; ++g) c[g] = MMUL(bias_tile(bias + (og + g) * 8));
      for (int kk = 0; kk < KT; ++kk) {
        v64 a = a_get(t, kk);
        _Pragma("clang loop unroll(full)")
        for (int g = 0; g < G; ++g) c[g].mac(a, aie::load_v<64>(w + ((og + g) * KT + kk) * 64));
      }
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < G; ++g) epi(og + g, t, c[g]);
    }
  }
}

inline void set_modes() {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
}

// A tile of 8 output pixels gathered from a [blocks][pixels][8] source with
// pixel index map `src(o)`; rows past `valid` are zero.
template <typename Map>
inline v64 gather_tile(const int8_t *base, int block_stride, int icb, int t, int valid, Map src) {
  alignas(64) int8_t tile[64];
  for (int r = 0; r < 8; ++r) {
    const int o = t * 8 + r;
    if (o < valid) copy8(tile + r * 8, base + (icb * block_stride + src(o)) * 8);
    else __builtin_memset(tile + r * 8, 0, 8);
  }
  return aie::load_v<64>(tile);
}
}  // namespace

#ifdef BLK_CONV1
extern "C" void fused_bottleneck_conv1_chunk(const int8_t *input, const uint8_t *params, uint8_t *bundle, int32_t chunk) {
  set_modes();
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS1_OFFSET);
  int8_t *out = (int8_t *)bundle;
  const v64 zero = aie::zeros<int8, 64>();
  if (chunk == 0) {
    constexpr int bytes = MB * PADP * 8;
    for (int i = 0; i + 64 <= bytes; i += 64) aie::store_unaligned_v(out + i, zero);
    for (int i = bytes - bytes % 64; i < bytes; ++i) out[i] = 0;
  }
  // Rows past P (P < 8 or a partial tail) read neighbouring bytes and are ignored.
  auto a_get = [&](int t, int icb) -> v64 { return flip(load_tile(input + (icb * P + t * 8) * 8)); };
  auto epi = [&](int ocl, int t, MMUL &c) {
    v64 v = aie::max(c.to_vector<int8>(SHIFT1_V), zero);
    const int gb = chunk * NB1 + ocl;
    if constexpr (ROW_TILES1) {
      const int y = (t * 8) / W, x0 = (t * 8) % W;
      aie::store_unaligned_v(out + (gb * PADP + (y + 1) * PW + x0 + 1) * 8, v);
    } else {
      alignas(64) int8_t scratch[64];
      aie::store_v(scratch, v);
      for (int r = 0; r < 8; ++r) {
        const int o = t * 8 + r;
        if (o < P) copy8(out + (gb * PADP + (o / W + 1) * PW + (o % W) + 1) * 8, scratch + r * 8);
      }
    }
  };
  tiled_gemm<pick_group(NB1)>(NB1, T1, CB, weights, bias, a_get, epi);
}
#endif

#if defined(BLK_CONV2A) || defined(BLK_CONV2B)
// Non-row-tiled maps: the 3x3 windows are gathered once per block (chunk 0) into a
// persistent im2col buffer of aligned A tiles, so every chunk's inner loop is a plain
// 64-byte load instead of eight scalar copies per MMUL.
alignas(64) static int8_t col_tiles[ROW_TILES2 ? 64 : NTAPS * MB * TO * 64];

static void conv2_impl(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk) {
  set_modes();
  const int8_t *in = (const int8_t *)bundle;
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS2_OFFSET);
  int8_t *out = (int8_t *)output;
  const v64 zero = aie::zeros<int8, 64>();
  if constexpr (!ROW_TILES2) {
    if (chunk == 0) {
      for (int vt = 0; vt < NTAPS; ++vt) {
        const int tap = TAPS.tap[vt], ky = tap / 3, kx = tap % 3;
        for (int icb = 0; icb < MB; ++icb)
          for (int t = 0; t < TO; ++t) {
            v64 tile = gather_tile(in, PADP, icb, t, OP, [&](int o) { return ((o / OW) * S + ky) * PW + (o % OW) * S + kx; });
            aie::store_v(col_tiles + ((vt * MB + icb) * TO + t) * 64, tile);
          }
      }
    }
  }
  auto a_get = [&](int t, int kk) -> v64 {
    if constexpr (ROW_TILES2) {
      const int tap = TAPS.tap[kk / MB], icb = kk % MB;
      const int ky = tap / 3, kx = tap % 3;
      const int y = (t * 8) / OW, x0 = (t * 8) % OW;
      return load_tile(in + (icb * PADP + (y + ky) * PW + x0 + kx) * 8);
    } else {
      return aie::load_v<64>(col_tiles + (kk * TO + t) * 64);
    }
  };
  auto epi = [&](int ocl, int t, MMUL &c) {
    v64 v = aie::max(c.to_vector<int8>(SHIFT2_V), zero);
    const int gb = chunk * NB2 + ocl;
    store_rows(out + (gb * OP + t * 8) * 8, v, OP - t * 8);
  };
  tiled_gemm<pick_group(NB2)>(NB2, TO, NTAPS * MB, weights, bias, a_get, epi);
}
#endif

#ifdef BLK_CONV2A
extern "C" void fused_bottleneck_conv2_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk, int32_t) {
  conv2_impl(bundle, params, output, chunk);
}
#endif
#ifdef BLK_CONV2B
extern "C" void fused_bottleneck_conv2_chunk_b(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk, int32_t) {
  conv2_impl(bundle, params, output, chunk);
}
#endif

#ifdef BLK_SKIP
constexpr bool SKIP_DIRECT = (SS == 1 && OP == P);
alignas(64) static int8_t skip_x[SKIP_DIRECT ? 64 : CB * TO * 64];

extern "C" void fused_bottleneck_skip_chunk(const int8_t *input, const uint8_t *params, uint8_t *output, int32_t chunk) {
  set_modes();
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_SKIP_BIAS_OFFSET);
  int8_t *out = (int8_t *)output;
  if constexpr (!SKIP_DIRECT) {
    // Strided projection: gather the sampled input pixels once per block.
    if (chunk == 0)
      for (int icb = 0; icb < CB; ++icb)
        for (int t = 0; t < TO; ++t)
          aie::store_v(skip_x + (icb * TO + t) * 64,
                       gather_tile(input, P, icb, t, OP, [&](int o) { return (o / OW) * SS * W + (o % OW) * SS; }));
  }
  auto a_get = [&](int t, int icb) -> v64 {
    if constexpr (SKIP_DIRECT) return flip(load_tile(input + (icb * P + t * 8) * 8));
    else return flip(aie::load_v<64>(skip_x + (icb * TO + t) * 64));
  };
  auto epi = [&](int ocl, int t, MMUL &c) {
    const int gb = chunk * NBS + ocl;
    store_rows(out + (gb * OP + t * 8) * 8, c.to_vector<int8>(SKIPSHIFT_V), OP - t * 8);
  };
  tiled_gemm<pick_group(NBS)>(NBS, TO, CB, weights, bias, a_get, epi);
}
#endif

#ifdef BLK_IDENTITY
extern "C" void fused_bottleneck_identity_skip(const int8_t *input, int8_t *output) {
  constexpr int bytes = P * FUSED_C;
  for (int i = 0; i + 64 <= bytes; i += 64) aie::store_unaligned_v(output + i, flip(load_tile(input + i)));
  for (int i = bytes - bytes % 64; i < bytes; ++i) output[i] = (int8_t)(input[i] ^ 0x80);
}
#endif

#ifdef BLK_CONV3
extern "C" void fused_bottleneck_conv3_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk) {
  set_modes();
  const int8_t *in = (const int8_t *)bundle;
  const int8_t *skip = in + OP * FUSED_MID;
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS3_OFFSET);
  int8_t *out = (int8_t *)output;
  const int EA = EA_V, EB = EB_V;
  const int common = (EA < EB) ? pos(-EA) : pos(-EB);
  const v64 zero = aie::zeros<int8, 64>();
  auto a_get = [&](int t, int icb) -> v64 { return load_tile(in + (icb * OP + t * 8) * 8); };
  auto epi = [&](int ocl, int t, MMUL &c) {
    const int gb = chunk * NB3 + ocl;
    v64 q3 = c.to_vector<int8>(SHIFT3_V);
    v64 r = load_tile(skip + (gb * OP + t * 8) * 8);
    aie::accum<acc32, 64> a1, a2;
    a1.from_vector(q3, EA + common);
    a2.from_vector(r, EB + common);
    aie::accum<acc32, 64> sum = aie::add(a1, a2);
    v64 v = aie::max(sum.to_vector<int8>(common), zero);
#if defined(FUSED_DBG) && FUSED_DBG == 1
    v = r;
#elif defined(FUSED_DBG) && FUSED_DBG == 2
    v = q3;
#endif
    store_rows(out + (gb * OP + t * 8) * 8, flip(v), OP - t * 8);
  };
  tiled_gemm<pick_group(NB3)>(NB3, TO, MB, weights, bias, a_get, epi);
}
#endif
