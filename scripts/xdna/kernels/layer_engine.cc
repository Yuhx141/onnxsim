// Layer-sequential engine kernel: one conv layer (or one K-chunk of it) on ONE core.
//
// A layer is spread over all 32 cores of the array: core `s` owns NBC consecutive 8-channel
// output blocks. Every weight chunk starts with a 192-byte descriptor (same idea as
// fused_bottleneck_rt.cc) so a single compiled kernel serves every layer shape.
//
// Activation layout between layers: the *producer's* per-core output objects are placed back to
// back, one REGION_BYTES region per producing core: region s = [local block][pixel][8] bytes. A
// consumer therefore sees its input as NCP regions of NBP blocks of P pixels; the reduction runs
// over (tap, region, local block) items, and the packed weights follow exactly that order.
//
// Chunk = a range of `tt` (tap x region) reduction steps; partial int32 sums stay in a static
// buffer between chunks of one layer (first chunk starts from the bias, the last one requantizes).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ENG_REGION_BYTES
#define ENG_REGION_BYTES 512
#endif
#ifndef ENG_ACC_TILES
#define ENG_ACC_TILES 8
#endif
#ifndef ENG_ACT_BYTES
#define ENG_ACT_BYTES 16384  // size of one activation object; its unused tail hosts the padded copy of a row-tiled 3x3 input
#endif
#ifndef ENG_SCRATCH_BLOCKS
#define ENG_SCRATCH_BLOCKS 8
#endif

namespace {
constexpr int DESC_BYTES = 192;
enum {
  D_NBP,      // blocks per input region (inner reduction length)
  D_NCP,      // input regions
  D_W,        // input map width / height
  D_H,
  D_OW,       // output map
  D_OH,
  D_S,        // stride of the 3x3 (or strided 1x1)
  D_MODE,     // 0 = direct 1x1, 1 = gather (3x3 / strided)
  D_NTAPS,
  D_TAP0,     // 9 words
  D_NB = D_TAP0 + 9,  // output blocks this core computes (0 = idle)
  D_TT0,      // first (tap,region) step in this chunk
  D_TTN,      // steps in this chunk
  D_FIRST,
  D_LAST,
  D_SHIFT,
  D_RELU,
  D_IN_FLIP,
  D_OUT_FLIP,
  D_RES,      // 0 none, 1 int8 residual, 2 uint8 (flip) residual
  D_EA,
  D_EB,
  D_BIAS,     // byte offset of the int32 bias inside the payload
  D_CORE,     // global core index (region index of this core's output/residual)
};

constexpr int pos(int v) { return v > 0 ? v : 0; }

using MMUL = aie::mmul<8, 8, 8, int8, int8>;
using v64 = aie::vector<int8, 64>;

alignas(64) static int32_t acc_buf[ENG_ACC_TILES * 64];

inline aie::vector<int32, 64> bias_tile(const int32_t *b) {
  aie::vector<int32, 8> v = aie::load_unaligned_v<8>(b);
  aie::vector<int32, 16> v2 = aie::concat(v, v);
  aie::vector<int32, 32> v4 = aie::concat(v2, v2);
  return aie::concat(v4, v4);
}

inline void copy8(int8_t *dst, const int8_t *src) { __builtin_memcpy(dst, src, 8); }

inline void store_rows(int8_t *dst, v64 v, int rows) {
  if (rows >= 8) {
    aie::store_unaligned_v(dst, v);
  } else {
    alignas(64) int8_t s[64];
    aie::store_v(s, v);
    for (int r = 0; r < rows; ++r) copy8(dst + r * 8, s + r * 8);
  }
}

struct Layer {
  const int32_t *d;
  const int8_t *weights;
  const int32_t *bias;
  int nbp, ncp, w, h, ow, oh, s, mode, ntaps, nb, tt0, ttn, first, last, shift, relu, in_flip, out_flip, res, ea, eb, core;
  int op, t_out;
  int taps[9];
};

inline Layer load(const uint8_t *slot) {
  Layer l;
  l.d = (const int32_t *)slot;
  const int32_t *d = l.d;
  l.weights = (const int8_t *)(slot + DESC_BYTES);
  l.nbp = d[D_NBP]; l.ncp = d[D_NCP]; l.w = d[D_W]; l.h = d[D_H]; l.ow = d[D_OW]; l.oh = d[D_OH];
  l.s = d[D_S]; l.mode = d[D_MODE]; l.ntaps = d[D_NTAPS];
  for (int i = 0; i < 9; ++i) l.taps[i] = d[D_TAP0 + i];
  l.nb = d[D_NB]; l.tt0 = d[D_TT0]; l.ttn = d[D_TTN]; l.first = d[D_FIRST]; l.last = d[D_LAST];
  l.shift = pos(d[D_SHIFT]); l.relu = d[D_RELU]; l.in_flip = d[D_IN_FLIP]; l.out_flip = d[D_OUT_FLIP];
  l.res = d[D_RES]; l.ea = d[D_EA]; l.eb = d[D_EB]; l.core = d[D_CORE];
  l.bias = (const int32_t *)((const uint8_t *)l.weights + d[D_BIAS]);
  l.op = l.ow * l.oh;
  l.t_out = (l.op + 7) / 8;
  return l;
}

// One 8-pixel A tile per input block of one region, gathered from per-row source pixel offsets
// (invalid rows are zeroed by a mask, so the copy is branch-free) into scratch_tiles.
inline void gather_region(int8_t *scratch, const int8_t *region, int nbp, int p_in, const int *offs, const uint64_t *mask) {
  uint64_t *dst = (uint64_t *)scratch;
  for (int l = 0; l < nbp; ++l) {
    const int8_t *base = region + l * p_in * 8;
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 8; ++r) dst[l * 8 + r] = *(const uint64_t *)(base + offs[r] * 8) & mask[r];
  }
}

template <int G, typename ABase, typename Epi>
inline void tiled_gemm(const Layer &L, int nt, ABase a_base, int a_stride, Epi epi) {
  const int nb = L.nb, KT = L.ttn * L.nbp;
  const int total_kt = L.ncp * L.ntaps * L.nbp;
  const v64 flipmask = aie::broadcast<int8, 64>(L.in_flip ? (int8_t)-128 : (int8_t)0);
  for (int t = 0; t < nt; ++t) {
    for (int og = 0; og < nb; og += G) {
      MMUL c[G];
      const int8_t *wrow[G];
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < G; ++g) {
        if (L.first) c[g] = MMUL(bias_tile(L.bias + (og + g) * 8));
        else c[g] = MMUL(aie::load_v<64>(acc_buf + ((og + g) * nt + t) * 64));
        wrow[g] = L.weights + (size_t)(og + g) * KT * 64;
      }
      (void)total_kt;
      for (int tt = 0; tt < L.ttn; ++tt) {
        const int8_t *ap = a_base(t, L.tt0 + tt);
        for (int icb = 0; icb < L.nbp; ++icb) {
          v64 a = aie::bit_xor(aie::load_unaligned_v<64>(ap), flipmask);
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
        if (og + g < nb) epi(og + g, t, c[g]);
    }
  }
}
}  // namespace

extern "C" void layer_chunk(const int8_t *act, const uint8_t *slot, int8_t *out, const int8_t *resid) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  const Layer L = load(slot);
  if (L.nb == 0) return;
  const int p_in = L.w * L.h;
  const v64 zero = aie::zeros<int8, 64>();

  int py[64], px[64];
  {
    int y = 0, x = 0;
    for (int o = 0; o < L.op; ++o) {
      py[o] = y; px[o] = x;
      if (++x == L.ow) { x = 0; ++y; }
    }
  }
  auto epi = [&](int ocl, int t, MMUL &c) __attribute__((noinline)) {
    if (!L.last) {
      aie::store_v(acc_buf + (ocl * L.t_out + t) * 64, c.to_vector<int32>(0));
      return;
    }
    v64 q = c.to_vector<int8>(L.shift);
    if (L.res) {
      v64 r = aie::load_unaligned_v<64>(resid + L.core * ENG_REGION_BYTES + (ocl * L.op + t * 8) * 8);
      if (L.res == 2) r = aie::bit_xor(r, aie::broadcast<int8, 64>((int8_t)-128));
      const int common = (L.ea < L.eb) ? pos(-L.ea) : pos(-L.eb);
      aie::accum<acc32, 64> a1, a2;
      a1.from_vector(q, L.ea + common);
      a2.from_vector(r, L.eb + common);
      aie::accum<acc32, 64> sum = aie::add(a1, a2);
      q = sum.to_vector<int8>(common);
    }
    if (L.relu) q = aie::max(q, zero);
    if (L.out_flip) q = aie::bit_xor(q, aie::broadcast<int8, 64>((int8_t)-128));
    store_rows(out + (ocl * L.op + t * 8) * 8, q, L.op - t * 8);
  };

  if (L.mode == 0) {
    // Direct: tile t of every input block is the 64 contiguous bytes at t*64; regions are REGION_BYTES apart.
    auto a_base = [&](int t, int tt) { return act + tt * ENG_REGION_BYTES + t * 64; };
    if (L.nb % 4 == 0) tiled_gemm<4>(L, L.t_out, a_base, p_in * 8, epi);
    else tiled_gemm<2>(L, L.t_out, a_base, p_in * 8, epi);
  } else {
    const bool row_tiles = (L.s == 1 && L.w % 8 == 0 && L.ntaps > 1 && L.ncp * ENG_REGION_BYTES + L.nbp * L.ncp * (L.h + 2) * (L.w + 2) * 8 <= ENG_ACT_BYTES);
    if (row_tiles) {
      // Padded copy of the input built once (first chunk); each A tile is then one unaligned row load.
      const int pw = L.w + 2, padp = (L.h + 2) * pw, nblk = L.nbp * L.ncp;
      int8_t *pad_buf = (int8_t *)act + L.ncp * ENG_REGION_BYTES;  // the object is ours while held; its tail is unused
      if (L.first) {
        const v64 z = aie::zeros<int8, 64>();
        for (int i = 0; i < nblk * padp * 8; i += 64) aie::store_v(pad_buf + i, z);
        for (int cp = 0; cp < L.ncp; ++cp)
          for (int l = 0; l < L.nbp; ++l) {
            const int8_t *src = act + cp * ENG_REGION_BYTES + l * p_in * 8;
            int8_t *dst = pad_buf + (cp * L.nbp + l) * padp * 8;
            for (int y = 0; y < L.h; ++y)
              for (int x = 0; x < L.w; ++x)
                *(uint64_t *)(dst + ((y + 1) * pw + x + 1) * 8) = *(const uint64_t *)(src + (y * L.w + x) * 8);
          }
      }
      auto a_base = [&](int t, int tt) {
        int ti = 0, cp = tt;
        while (cp >= L.ncp) { cp -= L.ncp; ++ti; }
        const int tap = L.taps[ti], ky = tap >= 6 ? 2 : (tap >= 3 ? 1 : 0), kx = tap - ky * 3;
        return (const int8_t *)pad_buf + (cp * L.nbp * padp + (py[t * 8] + ky) * pw + px[t * 8] + kx) * 8;
      };
      if (L.nb % 4 == 0) tiled_gemm<4>(L, L.t_out, a_base, padp * 8, epi);
      else tiled_gemm<2>(L, L.t_out, a_base, padp * 8, epi);
      return;
    }
    // Gather: step tt = tap_index * NCP + region. The eight source offsets/masks depend only on
    // (tap, pixel tile), so they are recomputed only when that pair changes (tt runs region-fastest).
    int offs[8];
    uint64_t mask[8];
    int cache_ti = -1, cache_t = -1;
    int8_t *scratch = (int8_t *)act + L.ncp * ENG_REGION_BYTES;  // the unused tail of the activation object
    auto a_base = [&](int t, int tt) {
      int ti = 0, cp = tt;
      while (cp >= L.ncp) { cp -= L.ncp; ++ti; }  // tiny loop: at most 8 iterations, no divide
      if (ti != cache_ti || t != cache_t) {
        cache_ti = ti; cache_t = t;
        const int tap = L.taps[ti], ky = tap >= 6 ? 2 : (tap >= 3 ? 1 : 0), kx = tap - ky * 3;
        for (int r = 0; r < 8; ++r) {
          const int o = t * 8 + r;
          const int oo = o < L.op ? o : 0;
          const int iy = py[oo] * L.s + ky - 1;  // 3x3 pad 1; a strided 1x1 is the centre tap (4)
          const int ix = px[oo] * L.s + kx - 1;
          const bool ok = iy >= 0 && iy < L.h && ix >= 0 && ix < L.w;
          offs[r] = ok ? iy * L.w + ix : 0;
          mask[r] = ok ? ~0ull : 0ull;
        }
      }
      gather_region(scratch, act + cp * ENG_REGION_BYTES, L.nbp, p_in, offs, mask);
      return (const int8_t *)scratch;
    };
    if (L.nb % 4 == 0) tiled_gemm<4>(L, L.t_out, a_base, 64, epi);
    else tiled_gemm<2>(L, L.t_out, a_base, 64, epi);
  }
}
