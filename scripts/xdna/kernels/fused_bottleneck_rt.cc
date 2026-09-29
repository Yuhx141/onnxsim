// Runtime-shaped ResNet bottleneck kernels.
//
// Same math, layouts and requantization as fused_bottleneck_blocked.cc (bit-identical), but the
// block geometry is NOT compiled in: every chunk of the weight stream starts with a 192-byte
// descriptor (see blocked_stage.RT_DESC_WORDS) that the kernels read at entry. One compiled
// kernel set therefore serves any block shape, which keeps program memory small (16 KB/core)
// and is the building block for layer-sequential engines where one core runs many layer shapes.
//
// Slot layout: [descriptor: 192 B][weight tiles ...][bias int32 ...]; bias offsets (relative to
// the tile area) are descriptor fields. Only buffer *capacities* stay compile-time:
//   RT_COL_BYTES   static im2col buffer (conv2 cores)      RT_SKIPX_BYTES  strided-skip gather (conv1 core)
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef RT_COL_BYTES
#define RT_COL_BYTES 64
#endif
#ifndef RT_SKIPX_BYTES
#define RT_SKIPX_BYTES 64
#endif

namespace {
constexpr int DESC_BYTES = 192;
// Descriptor word indices (int32).
enum {
  D_SHIFT1, D_SHIFT2, D_SHIFT3, D_SKIPSHIFT, D_EA, D_EB,
  D_W, D_H, D_C, D_MID, D_OUTC, D_OW, D_OH, D_S, D_SS,
  D_C1CH, D_C2CH, D_C3CH, D_SKCH, D_NTAPS, D_TAP0 /* 9 words */ = 20,
  D_BIAS1 = 29, D_BIAS2, D_BIAS3, D_SKBIAS,
  D_NB1 = 33, D_NB2, D_NB3, D_NBS,  // output blocks per chunk, precomputed by the host (no divide on the core)
};

constexpr int pos(int v) { return v > 0 ? v : 0; }

struct Dims {
  int W, H, P, OW, OH, OP, S, SS, PW, PADP;
  int CB, MB, HB, OB, T1, TO, NB1, NB2, NB3, NBS, NTAPS;
  int sh1, sh2, sh3, shs, ea, eb;
  int taps[9];
  int b1, b2, b3, bs;
  const int8_t *weights;
};

inline Dims load_dims(const uint8_t *params) {
  const int32_t *d = (const int32_t *)params;
  Dims x;
  x.W = d[D_W]; x.H = d[D_H]; x.P = x.W * x.H;
  x.OW = d[D_OW]; x.OH = d[D_OH]; x.OP = x.OW * x.OH;
  x.S = d[D_S]; x.SS = d[D_SS];
  x.PW = x.W + 2; x.PADP = (x.H + 2) * x.PW;
  x.CB = d[D_C] / 8; x.MB = d[D_MID] / 8; x.HB = x.MB / 2; x.OB = d[D_OUTC] / 8;
  x.T1 = (x.P + 7) / 8; x.TO = (x.OP + 7) / 8;
  x.NB1 = d[D_NB1]; x.NB2 = d[D_NB2]; x.NB3 = d[D_NB3]; x.NBS = d[D_NBS];
  x.NTAPS = d[D_NTAPS];
  for (int i = 0; i < 9; ++i) x.taps[i] = d[D_TAP0 + i];
  x.sh1 = pos(d[D_SHIFT1]); x.sh2 = pos(d[D_SHIFT2]); x.sh3 = pos(d[D_SHIFT3]); x.shs = pos(d[D_SKIPSHIFT]);
  x.ea = d[D_EA]; x.eb = d[D_EB];
  x.b1 = d[D_BIAS1]; x.b2 = d[D_BIAS2]; x.b3 = d[D_BIAS3]; x.bs = d[D_SKBIAS];
  x.weights = (const int8_t *)(params + DESC_BYTES);
  return x;
}

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

inline void store_rows(int8_t *dst, v64 v, int rows) {
  if (rows >= 8) {
    aie::store_unaligned_v(dst, v);
  } else {
    alignas(64) int8_t scratch[64];
    aie::store_v(scratch, v);
    for (int r = 0; r < rows; ++r) copy8(dst + r * 8, scratch + r * 8);
  }
}

// GEMM over one weight chunk. The reduction is NT taps x KB input blocks (NT = 1 for 1x1 convs).
// a_base(t, tt) returns the address of the A tile for input block 0 and `a_stride` is the byte
// distance between consecutive input blocks, so the inner loop is one load, one pointer add and
// G MACs: no index arithmetic, and (there being no hardware integer divide on the AIE) no division.
// FLIP xors the loaded activation with 0x80 (uint8-with-zero-point-128 -> int8)
// (all A loads are unaligned-capable so one instantiation serves aligned and unaligned sources).
template <int G, bool FLIP, typename ABase, typename Epi>
inline void tiled_gemm(int nb, int T, int NT, int KB, const int8_t *w, const int32_t *bias, ABase a_base, int a_stride,
                       Epi epi) {
  const int KT = NT * KB;
  for (int t = 0; t < T; ++t) {
    for (int og = 0; og < nb; og += G) {
      MMUL c[G];
      const int8_t *wrow[G];
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < G; ++g) {
        c[g] = MMUL(bias_tile(bias + (og + g) * 8));
        wrow[g] = w + (size_t)(og + g) * KT * 64;
      }
      for (int tt = 0; tt < NT; ++tt) {
        const int8_t *ap = a_base(t, tt);
        _Pragma("clang loop min_iteration_count(4)")
        for (int icb = 0; icb < KB; ++icb) {
          v64 a = load_tile(ap);
          if (FLIP) a = flip(a);
          ap += a_stride;
          _Pragma("clang loop unroll(full)")
          for (int g = 0; g < G; ++g) {
            c[g].mac(a, aie::load_v<64>(wrow[g]));
            wrow[g] += 64;
          }
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < G; ++g)
        if (og + g < nb) epi(og + g, t, c[g]);  // guard: odd block counts run a G=2 loop with a dead tail
    }
  }
}

// Two instantiations only (program memory is 16 KB/core); odd counts use G=2 with a guarded tail.
template <bool FLIP, typename ABase, typename Epi>
inline void gemm_rt(int nb, int T, int NT, int KB, const int8_t *w, const int32_t *bias, ABase a_base, int a_stride,
                    Epi epi) {
  if (nb % 4 == 0) tiled_gemm<4, FLIP>(nb, T, NT, KB, w, bias, a_base, a_stride, epi);
  else tiled_gemm<2, FLIP>(nb, T, NT, KB, w, bias, a_base, a_stride, epi);
}

// Row/column of every pixel of a w-wide map, by incremental counters (no division: the AIE has no
// hardware divider, and these tables feed the gather/scatter paths that would otherwise divide per pixel).
inline void fill_coords(int *py, int *px, int n, int w) {
  int y = 0, x = 0;
  for (int o = 0; o < n; ++o) {
    py[o] = y;
    px[o] = x;
    if (++x == w) { x = 0; ++y; }
  }
}

// Copy one A tile (8 pixel rows of 8 bytes) with per-row source pixel offsets `offs` (< 0 = padding
// row -> zeros). Sources and destination are 8-byte aligned, so each row is one 64-bit access.
inline void gather_rows(int8_t *dst, const int8_t *block_base, const int *offs) {
  uint64_t *d = (uint64_t *)dst;
  for (int r = 0; r < 8; ++r) d[r] = offs[r] >= 0 ? *(const uint64_t *)(block_base + offs[r] * 8) : 0;
}

inline void set_modes() {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
}

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
  const Dims d = load_dims(params);
  const int32_t *bias = (const int32_t *)((const uint8_t *)d.weights + d.b1);
  int8_t *out = (int8_t *)bundle;
  const v64 zero = aie::zeros<int8, 64>();
  if (chunk == 0) {
    const int bytes = d.MB * d.PADP * 8;
    for (int i = 0; i + 64 <= bytes; i += 64) aie::store_unaligned_v(out + i, zero);
    for (int i = bytes - bytes % 64; i < bytes; ++i) out[i] = 0;
  }
  const bool row_tiles = (d.W % 8 == 0);
  int py[64], px[64];  // per-pixel row/column (P <= 64), computed once without division
  fill_coords(py, px, d.P, d.W);
  int ty[8], tx[8];    // per-tile row/column of the first pixel
  for (int t = 0; t < d.T1 && t < 8; ++t) { ty[t] = py[t * 8]; tx[t] = px[t * 8]; }
  auto a_base = [&](int t, int) { return input + t * 64; };
  auto epi = [&](int ocl, int t, MMUL &c) __attribute__((noinline)) {
    v64 v = aie::max(c.to_vector<int8>(d.sh1), zero);
    const int gb = chunk * d.NB1 + ocl;
    if (row_tiles) {
      aie::store_unaligned_v(out + (gb * d.PADP + (ty[t] + 1) * d.PW + tx[t] + 1) * 8, v);
    } else {
      alignas(64) int8_t scratch[64];
      aie::store_v(scratch, v);
      for (int r = 0; r < 8; ++r) {
        const int o = t * 8 + r;
        if (o < d.P) copy8(out + (gb * d.PADP + (py[o] + 1) * d.PW + px[o] + 1) * 8, scratch + r * 8);
      }
    }
  };
  gemm_rt<true>(d.NB1, d.T1, 1, d.CB, d.weights, bias, a_base, d.P * 8, epi);
}
#endif

#if defined(BLK_CONV2A) || defined(BLK_CONV2B)
alignas(64) static int8_t col_tiles[RT_COL_BYTES];

static void conv2_impl(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk) {
  set_modes();
  const Dims d = load_dims(params);
  const int8_t *in = (const int8_t *)bundle;
  const int32_t *bias = (const int32_t *)((const uint8_t *)d.weights + d.b2);
  int8_t *out = (int8_t *)output;
  const v64 zero = aie::zeros<int8, 64>();
  const bool row_tiles = (d.S == 1 && d.W % 8 == 0 && d.OW == d.W);
  int py[64], px[64];
  fill_coords(py, px, d.OP, d.OW);
  int ty[8], tx[8];
  for (int t = 0; t < d.TO && t < 8; ++t) { ty[t] = py[t * 8]; tx[t] = px[t * 8]; }
#ifdef RT_SKIP_GATHER
  if (false) {  // profiling only: skip the im2col build (results are wrong)
#else
  if (!row_tiles && chunk == 0) {
#endif
    // Build every (tap, pixel tile, input block) window once per block into static im2col tiles.
    // The eight source pixel offsets depend only on (tap, tile), so compute them once and copy
    // the same rows for every input block.
    for (int vt = 0; vt < d.NTAPS; ++vt) {
      const int tap = d.taps[vt], ky = tap / 3, kx = tap - ky * 3;
      for (int t = 0; t < d.TO; ++t) {
        int offs[8];
        for (int r = 0; r < 8; ++r) {
          const int o = t * 8 + r;
          offs[r] = o < d.OP ? (py[o] * d.S + ky) * d.PW + px[o] * d.S + kx : -1;
        }
        for (int icb = 0; icb < d.MB; ++icb)
          gather_rows(col_tiles + ((vt * d.MB + icb) * d.TO + t) * 64, in + (size_t)icb * d.PADP * 8, offs);
      }
    }
  }
  // Row-tiled maps read unaligned 8-pixel rows straight from the padded buffer; other maps read
  // the static im2col tiles built above (aligned, one tile per (tap, input block, pixel tile)).
  auto a_base_row = [&](int t, int tt) {
    const int tap = d.taps[tt];
    const int ky = tap / 3, kx = tap - ky * 3;
    return in + ((ty[t] + ky) * d.PW + tx[t] + kx) * 8;
  };
  auto a_base_col = [&](int t, int tt) { return col_tiles + (tt * d.MB * d.TO + t) * 64; };
  auto epi = [&](int ocl, int t, MMUL &c) __attribute__((noinline)) {
    v64 v = aie::max(c.to_vector<int8>(d.sh2), zero);
    const int gb = chunk * d.NB2 + ocl;
    store_rows(out + (gb * d.OP + t * 8) * 8, v, d.OP - t * 8);
  };
  if (row_tiles) gemm_rt<false>(d.NB2, d.TO, d.NTAPS, d.MB, d.weights, bias, a_base_row, d.PADP * 8, epi);
  else gemm_rt<false>(d.NB2, d.TO, d.NTAPS, d.MB, d.weights, bias, a_base_col, d.TO * 64, epi);
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
alignas(64) static int8_t skip_x[RT_SKIPX_BYTES];

extern "C" void fused_bottleneck_skip_chunk(const int8_t *input, const uint8_t *params, uint8_t *output, int32_t chunk) {
  set_modes();
  const Dims d = load_dims(params);
  const int32_t *bias = (const int32_t *)((const uint8_t *)d.weights + d.bs);
  int8_t *out = (int8_t *)output;
  const bool direct = (d.SS == 1 && d.OP == d.P);
  int py[64], px[64];
  if (!direct && chunk == 0) fill_coords(py, px, d.OP, d.OW);
  if (!direct && chunk == 0)
    for (int t = 0; t < d.TO; ++t) {
      int offs[8];
      for (int r = 0; r < 8; ++r) {
        const int o = t * 8 + r;
        offs[r] = o < d.OP ? py[o] * d.SS * d.W + px[o] * d.SS : -1;
      }
      for (int icb = 0; icb < d.CB; ++icb)
        gather_rows(skip_x + (icb * d.TO + t) * 64, input + (size_t)icb * d.P * 8, offs);
    }
  auto a_base_direct = [&](int t, int) { return input + t * 64; };
  auto a_base_gather = [&](int t, int) { return skip_x + t * 64; };
  auto epi = [&](int ocl, int t, MMUL &c) __attribute__((noinline)) {
    const int gb = chunk * d.NBS + ocl;
    store_rows(out + (gb * d.OP + t * 8) * 8, c.to_vector<int8>(d.shs), d.OP - t * 8);
  };
  if (direct) gemm_rt<true>(d.NBS, d.TO, 1, d.CB, d.weights, bias, a_base_direct, d.P * 8, epi);
  else gemm_rt<true>(d.NBS, d.TO, 1, d.CB, d.weights, bias, a_base_gather, d.TO * 64, epi);
}
#endif

#ifdef BLK_IDENTITY
// bytes = P * C of the block input (a per-group constant passed by the design).
extern "C" void fused_bottleneck_identity_skip(const int8_t *input, int8_t *output, int32_t bytes) {
  for (int i = 0; i + 64 <= bytes; i += 64) aie::store_unaligned_v(output + i, flip(load_tile(input + i)));
  for (int i = bytes - bytes % 64; i < bytes; ++i) output[i] = (int8_t)(input[i] ^ 0x80);
}
#endif

#ifdef BLK_CONV3
extern "C" void fused_bottleneck_conv3_chunk(const uint8_t *bundle, const uint8_t *params, uint8_t *output, int32_t chunk) {
  set_modes();
  const Dims d = load_dims(params);
  const int8_t *in = (const int8_t *)bundle;
  const int8_t *skip = in + d.OP * d.MB * 8;
  const int32_t *bias = (const int32_t *)((const uint8_t *)d.weights + d.b3);
  int8_t *out = (int8_t *)output;
  const int EA = d.ea, EB = d.eb;
  const int common = (EA < EB) ? pos(-EA) : pos(-EB);
  const v64 zero = aie::zeros<int8, 64>();
  auto a_base = [&](int t, int) { return in + t * 64; };
  auto epi = [&](int ocl, int t, MMUL &c) __attribute__((noinline)) {
    const int gb = chunk * d.NB3 + ocl;
    v64 q3 = c.to_vector<int8>(d.sh3);
    v64 r = load_tile(skip + (gb * d.OP + t * 8) * 8);
    aie::accum<acc32, 64> a1, a2;
    a1.from_vector(q3, EA + common);
    a2.from_vector(r, EB + common);
    aie::accum<acc32, 64> sum = aie::add(a1, a2);
    v64 v = aie::max(sum.to_vector<int8>(common), zero);
    store_rows(out + (gb * d.OP + t * 8) * 8, flip(v), d.OP - t * 8);
  };
  gemm_rt<false>(d.NB3, d.TO, 1, d.MB, d.weights, bias, a_base, d.OP * 8, epi);
}
#endif
