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
#define ENG_SCRATCH_BLOCKS 4
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
  D_TI0,      // (tap index, region) decomposition of tt0, computed by the host (no divide on the core)
  D_CP0,
  D_REG,      // bytes between the input map's producing-core regions (512 unless the host wrote a wider layout)
  D_CLAMP,    // upper bound of the int8 result (127 = none): ReLU6 in the quantized domain
  D_KSZ,      // depthwise kernel size K (all K*K tap vectors are packed; padding (K-1)/2)
};

constexpr int pos(int v) { return v > 0 ? v : 0; }

using MMUL = aie::mmul<8, 8, 8, int8, int8>;
using v64 = aie::vector<int8, 64>;

alignas(64) static int8_t scratch_tiles[ENG_SCRATCH_BLOCKS * 64];
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
  int nbp, ncp, w, h, ow, oh, s, mode, ntaps, nb, tt0, ttn, first, last, shift, relu, in_flip, out_flip, res, ea, eb, core, ti0, cp0, clamp, reg, ksz;
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
  l.res = d[D_RES]; l.ea = d[D_EA]; l.eb = d[D_EB]; l.core = d[D_CORE]; l.ti0 = d[D_TI0]; l.cp0 = d[D_CP0]; l.clamp = d[D_CLAMP]; l.reg = d[D_REG]; l.ksz = d[D_KSZ];
  l.bias = (const int32_t *)((const uint8_t *)l.weights + d[D_BIAS]);
  l.op = l.ow * l.oh;
  l.t_out = (l.op + 7) / 8;
  return l;
}

// One 8-pixel A tile per input block of one region, gathered from per-row source pixel offsets
// (invalid rows are zeroed by a mask, so the copy is branch-free) into scratch_tiles.
inline void gather_region(int8_t *scratch, const int8_t *region, int nbp, int p_in, const int *offs, const uint64_t *mask, uint64_t flip) {
  uint64_t *dst = (uint64_t *)scratch;
  for (int l = 0; l < nbp; ++l) {
    const int8_t *base = region + l * p_in * 8;
    _Pragma("clang loop unroll(full)")
#ifdef ENG_GATHER_NOMASK
    for (int r = 0; r < 8; ++r) dst[l * 8 + r] = *(const uint64_t *)(base + offs[r] * 8) ^ flip;
#else
    for (int r = 0; r < 8; ++r) dst[l * 8 + r] = (*(const uint64_t *)(base + offs[r] * 8) ^ flip) & mask[r];  // re-centre uint8 inputs before masking: padding must stay 0
#endif
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

// Round-half-to-even arithmetic right shift (k >= 0), the rounding the QuantizeLinear after a mean / product uses.
static inline int rshift_even(int v, int k) {
  if (k == 0) return v;
  const int r = v >> k, rem = v - (r << k), half = 1 << (k - 1);
  return (rem > half || (rem == half && (r & 1))) ? r + 1 : r;
}

// Scale `n` uint8-with-zero-point-128 bytes (n multiple of 8) by 2^e in place, saturating, 8 bytes at a time.
__attribute__((noinline)) static void requant_bytes(uint8_t *p, int n, int e) {
  const v64 flip = aie::broadcast<int8, 64>((int8_t)-128);
  for (int i = 0; i < n; i += 8) {
    alignas(64) int8_t tile[64];
    aie::store_v(tile, aie::zeros<int8, 64>());
    __builtin_memcpy(tile, p + i, 8);
    aie::accum<acc32, 64> a;
    a.from_vector(aie::bit_xor(aie::load_v<64>(tile), flip), e > 0 ? e : 0);
    aie::store_v(tile, aie::bit_xor(a.to_vector<int8>(e > 0 ? 0 : -e), flip));
    __builtin_memcpy(p + i, tile, 8);
  }
}

// dst = requant(a * 2^ea + b * 2^eb) for n bytes (multiple of 8) of uint8-with-zero-point-128 activations,
// the same arithmetic as the residual epilogue: both operands are left-shifted to a common exponent, summed and
// rounded back with one shift.
__attribute__((noinline)) static void add_bytes(uint8_t *dst, const uint8_t *a, const uint8_t *b, int n, int ea, int eb) {
  const v64 flip = aie::broadcast<int8, 64>((int8_t)-128);
  const int common = (ea < eb) ? pos(-ea) : pos(-eb);
  for (int i = 0; i < n; i += 8) {
    alignas(64) int8_t ta[64], tb[64];
    aie::store_v(ta, aie::zeros<int8, 64>());
    aie::store_v(tb, aie::zeros<int8, 64>());
    __builtin_memcpy(ta, a + i, 8);
    __builtin_memcpy(tb, b + i, 8);
    aie::accum<acc32, 64> a1, a2;
    a1.from_vector(aie::bit_xor(aie::load_v<64>(ta), flip), ea + common);
    a2.from_vector(aie::bit_xor(aie::load_v<64>(tb), flip), eb + common);
    aie::accum<acc32, 64> sum = aie::add(a1, a2);
    aie::store_v(ta, aie::bit_xor(sum.to_vector<int8>(common), flip));
    __builtin_memcpy(dst + i, ta, 8);
  }
}

extern "C" void layer_chunk(const int8_t *act, const uint8_t *slot, int8_t *out, const int8_t *resid) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  const Layer L = load(slot);
  if (L.nb == 0) return;
  if (L.mode == 6 || L.mode == 7 || L.mode == 8) {
    // Data-movement jobs over this core's blocks: 6 = copy/concat/split (per-block source and scale exponent in the
    // payload), 7 = nearest upsample by D_S, 8 = max pool (window D_NTAPS, stride D_S, "same" padding). The result
    // is optionally re-scaled by 2^D_EA (the Q after the op has its own scale).
    const int n = L.op * 8;
    uint8_t *dst0 = (uint8_t *)out;
    if (L.mode == 6) {
      const int32_t *tab = (const int32_t *)L.weights;
      for (int ol = 0; ol < L.nb; ++ol) {
        const uint8_t *src = (const uint8_t *)(tab[ol * 3] ? resid : act) + tab[ol * 3 + 1];
        uint8_t *dst = dst0 + ol * n;
        for (int i = 0; i < n; i += 8) *(uint64_t *)(dst + i) = *(const uint64_t *)(src + i);
        const int e = tab[ol * 3 + 2];
        if (e != 0) requant_bytes(dst, n, e);
      }
    } else if (L.mode == 7) {
      const int f = L.s;
      for (int ol = 0; ol < L.nb; ++ol) {
        const uint8_t *blk = (const uint8_t *)act + L.core * L.reg + ol * L.w * L.h * 8;
        uint8_t *dst = dst0 + ol * n;
        int iy = 0, ry = 0;
        for (int oy = 0; oy < L.oh; ++oy) {
          int ix = 0, rx = 0;
          for (int ox = 0; ox < L.ow; ++ox) {
            *(uint64_t *)(dst + (oy * L.ow + ox) * 8) = *(const uint64_t *)(blk + (iy * L.w + ix) * 8);
            if (++rx == f) { rx = 0; ++ix; }
          }
          if (++ry == f) { ry = 0; ++iy; }
        }
      }
    } else {
      const int k = L.ntaps, pad = (k - 1) / 2;
      for (int ol = 0; ol < L.nb; ++ol) {
        const uint8_t *blk = (const uint8_t *)act + L.core * L.reg + ol * L.w * L.h * 8;
        uint8_t *dst = dst0 + ol * n;
        for (int oy = 0; oy < L.oh; ++oy)
          for (int ox = 0; ox < L.ow; ++ox) {
            uint8_t best[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // padding is byte 0: the smallest u8 activation, i.e. -inf
            for (int ky = 0; ky < k; ++ky) {
              const int iy = oy * L.s + ky - pad;
              if (iy < 0 || iy >= L.h) continue;
              for (int kx = 0; kx < k; ++kx) {
                const int ix = ox * L.s + kx - pad;
                if (ix < 0 || ix >= L.w) continue;
                const uint8_t *p = blk + (iy * L.w + ix) * 8;
                for (int c = 0; c < 8; ++c) best[c] = p[c] > best[c] ? p[c] : best[c];
              }
            }
            for (int c = 0; c < 8; ++c) dst[(oy * L.ow + ox) * 8 + c] = best[c];
          }
      }
    }
    if (L.mode != 6 && L.ea != 0) requant_bytes(dst0, L.nb * n, L.ea);
    return;
  }
  if (L.mode == 10) {
    // Global average pool of this core's blocks (pixel count a power of two): out = sat(rshift_even(sum, D_SHIFT)).
    const int n = L.w * L.h;
    for (int ol = 0; ol < L.nb; ++ol) {
      const uint8_t *blk = (const uint8_t *)act + L.core * L.reg + ol * n * 8;
      int sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int p = 0; p < n; ++p)
        for (int c = 0; c < 8; ++c) sum[c] += (int)blk[p * 8 + c] - 128;
      for (int c = 0; c < 8; ++c) {
        int q = rshift_even(sum[c], L.shift);
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        ((uint8_t *)out)[ol * 8 + c] = (uint8_t)(q + 128);
      }
    }
    return;
  }
  if (L.mode == 11) {
    // Broadcast multiply (squeeze-excite): out = sat(rshift_even((a - 128) * (b_c - 128), D_SHIFT)), b = 1x1 map in resid.
    const int n = L.w * L.h;
    for (int ol = 0; ol < L.nb; ++ol) {
      const uint8_t *blk = (const uint8_t *)act + L.core * L.reg + ol * n * 8;
      const uint8_t *scale = (const uint8_t *)resid + L.core * L.tt0 + ol * 8;
      uint8_t *dst = (uint8_t *)out + ol * n * 8;
      for (int p = 0; p < n; ++p)
        for (int c = 0; c < 8; ++c) {
          int q = rshift_even(((int)blk[p * 8 + c] - 128) * ((int)scale[c] - 128), L.shift);
          q = q > 127 ? 127 : (q < -128 ? -128 : q);
          dst[p * 8 + c] = (uint8_t)(q + 128);
        }
    }
    return;
  }
  if (L.mode == 9) {
    // Add of two activations (same channel blocking): A = act, B = resid with region size D_TT0.
    const int n = L.w * L.h * 8;
    for (int ol = 0; ol < L.nb; ++ol)
      add_bytes((uint8_t *)out + ol * n, (const uint8_t *)act + L.core * L.reg + ol * n,
                (const uint8_t *)resid + L.core * L.tt0 + ol * n, n, L.ea, L.eb);
    return;
  }
  if (L.mode == 5) {
    // Unary lookup table over this core's blocks (same layout in and out): out byte = table[in byte]. Any
    // elementwise activation (SiLU, HardSwish, Sigmoid, GELU, Tanh, ...) that was quantized to 8 bits reduces to
    // one 256-entry table built on the host from the op's float definition.
    const uint8_t *tab = (const uint8_t *)L.weights;
    const uint8_t *src = (const uint8_t *)act + L.core * L.reg;
    uint8_t *dst = (uint8_t *)out;
    const int n = L.nb * L.w * L.h * 8;
    for (int i = 0; i < n; ++i) dst[i] = tab[src[i]];
    return;
  }
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
    if (L.clamp < 127) q = aie::min(q, aie::broadcast<int8, 64>((int8_t)L.clamp));
    if (L.out_flip) q = aie::bit_xor(q, aie::broadcast<int8, 64>((int8_t)-128));
    store_rows(out + (ocl * L.op + t * 8) * 8, q, L.op - t * 8);
  };

  if (L.mode == 4) {
    // Depthwise 3x3 / strided: every output block reads only its own input block (input regions and output
    // regions belong to the same core), one elementwise multiply-accumulate per tap with the channel weights
    // replicated across the 8 pixel rows of the tile. Tiles are gathered per tap (masked, flip-before-mask).
    const uint64_t flip64 = L.in_flip ? 0x8080808080808080ull : 0ull;
    int offs[8];
    uint64_t mask[8];
    for (int t = 0; t < L.t_out; ++t)
      for (int ol = 0; ol < L.nb; ++ol) {
        aie::accum<acc32, 64> acc;
        acc.from_vector(bias_tile(L.bias + ol * 8), 0);
        const int8_t *blk = act + L.core * L.reg + ol * p_in * 8;
        const int K = L.ksz, padk = (K - 1) / 2;
        int ti = 0;
        for (int ky = 0; ky < K; ++ky)
          for (int kx = 0; kx < K; ++kx, ++ti) {
            for (int r = 0; r < 8; ++r) {
              const int o = t * 8 + r;
              const int oo = o < L.op ? o : 0;
              const int iy = py[oo] * L.s + ky - padk;
              const int ix = px[oo] * L.s + kx - padk;
              const bool ok = iy >= 0 && iy < L.h && ix >= 0 && ix < L.w;
              offs[r] = ok ? iy * L.w + ix : 0;
              mask[r] = ok ? ~0ull : 0ull;
            }
            uint64_t *dst = (uint64_t *)scratch_tiles;
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 8; ++r) dst[r] = (*(const uint64_t *)(blk + offs[r] * 8) ^ flip64) & mask[r];
            acc = aie::mac(acc, aie::load_v<64>(scratch_tiles), aie::load_v<64>(L.weights + (ol * K * K + ti) * 64));
          }
        MMUL c(acc.to_vector<int32>(0));
        epi(ol, t, c);
      }
    return;
  }
  if (L.mode == 0) {
    // Direct: tile t of every input block is the 64 contiguous bytes at t*64; regions are REGION_BYTES apart.
    auto a_base = [&](int t, int tt) { return act + tt * L.reg + t * 64; };
#ifndef ENG_NO_G4  // graph-compiled nets rarely need 4-block groups and the code space is tight
    if (L.nb % 4 == 0) tiled_gemm<4>(L, L.t_out, a_base, p_in * 8, epi);
    else
#endif
    tiled_gemm<2>(L, L.t_out, a_base, p_in * 8, epi);
  } else {
    const int pad_w = L.w + 2, pad_bytes = (L.h + 2) * pad_w * 8;
    const bool pad_fits = L.nbp * (L.w * L.h * 8 + pad_bytes) <= L.reg ||
                          L.ncp * L.reg + L.nbp * L.ncp * pad_bytes <= ENG_ACT_BYTES;
    if (L.s == 1 && pad_fits) {
      // Stride-1 3x3: a zero-padded copy of every input block is built once (first chunk); an A tile is
      // then 1/2/4 contiguous row segments of that copy, so no per-row gather is needed. The copy sits
      // in the unused bytes of each input region when it fits there, else in the activation object's tail.
      const int pw = L.w + 2, padp = (L.h + 2) * pw, lps = padp * 8;
      const bool in_region = L.nbp * (p_in + padp) * 8 <= L.reg;
      int8_t *pad0 = (int8_t *)act + (in_region ? L.nbp * p_in * 8 : L.ncp * L.reg);
      const int cps = in_region ? L.reg : L.nbp * lps;
      const int8_t fl = L.in_flip ? (int8_t)-128 : (int8_t)0;  // uint8 inputs are re-centred while copying, so the zero border stays 0
      if (L.first) {
        for (int cp = 0; cp < L.ncp; ++cp)
          for (int l = 0; l < L.nbp; ++l) {
            uint64_t *z = (uint64_t *)(pad0 + cp * cps + l * lps);
            for (int i = 0; i < padp; ++i) z[i] = 0;
            const int8_t *src = act + cp * L.reg + l * p_in * 8;
            int8_t *dst = pad0 + cp * cps + l * lps;
            for (int y = 0; y < L.h; ++y) {
              int8_t *d = dst + ((y + 1) * pw + 1) * 8;
              const int8_t *sp = src + y * L.w * 8;
              if (L.w >= 8) {
                for (int x = 0; x < L.w; x += 8) aie::store_unaligned_v(d + x * 8, aie::bit_xor(aie::load_unaligned_v<64>(sp + x * 8), aie::broadcast<int8, 64>(fl)));
              } else if (L.w == 4) {
                aie::store_unaligned_v(d, aie::bit_xor(aie::load_unaligned_v<32>(sp), aie::broadcast<int8, 32>(fl)));
              } else if (L.w == 2) {
                aie::store_unaligned_v(d, aie::bit_xor(aie::load_unaligned_v<16>(sp), aie::broadcast<int8, 16>(fl)));
              } else {
                *(uint64_t *)d = *(const uint64_t *)sp ^ (L.in_flip ? 0x8080808080808080ull : 0ull);
              }
            }
          }
      }
      const int seg = L.w >= 8 ? 1 : (L.w == 4 ? 2 : (L.w == 2 ? 4 : 1));
      const int rowb = pw * 8;
      for (int t = 0; t < L.t_out; ++t) {
        const int trow = (py[t * 8] * pw + px[t * 8]) * 8;
        for (int og = 0; og < L.nb; og += 2) {
          const int og1 = og + 1 < L.nb ? og + 1 : og;
          MMUL c0, c1;
          if (L.first) {
            c0 = MMUL(bias_tile(L.bias + og * 8));
            c1 = MMUL(bias_tile(L.bias + og1 * 8));
          } else {
            c0 = MMUL(aie::load_v<64>(acc_buf + (og * L.t_out + t) * 64));
            c1 = MMUL(aie::load_v<64>(acc_buf + (og1 * L.t_out + t) * 64));
          }
          const int8_t *wb0 = L.weights + (size_t)og * L.ttn * L.nbp * 64, *wb1 = L.weights + (size_t)og1 * L.ttn * L.nbp * 64;
          int ti = L.ti0, cp = L.cp0, remaining = L.ttn, done = 0;
          while (remaining > 0) {
            const int run = (L.ncp - cp) < remaining ? (L.ncp - cp) : remaining;
            const int tap = L.taps[ti], ky = tap >= 6 ? 2 : (tap >= 3 ? 1 : 0), kx = tap - ky * 3;
            const int8_t *base = pad0 + cp * cps + (ky * pw + kx) * 8 + trow;
            for (int l = 0; l < L.nbp; ++l) {
              const int8_t *ap = base + l * lps;
              const int8_t *w0 = wb0 + ((size_t)done * L.nbp + l) * 64, *w1 = wb1 + ((size_t)done * L.nbp + l) * 64;
              const int wstride = L.nbp * 64;
              if (seg == 1) {
                for (int i = 0; i < run; ++i) {
                  v64 a = aie::load_unaligned_v<64>(ap);
                  ap += cps;
                  c0.mac(a, aie::load_v<64>(w0)); w0 += wstride;
                  c1.mac(a, aie::load_v<64>(w1)); w1 += wstride;
                }
              } else if (seg == 2) {
                for (int i = 0; i < run; ++i) {
                  v64 a = aie::concat(aie::load_unaligned_v<32>(ap), aie::load_unaligned_v<32>(ap + rowb));
                  ap += cps;
                  c0.mac(a, aie::load_v<64>(w0)); w0 += wstride;
                  c1.mac(a, aie::load_v<64>(w1)); w1 += wstride;
                }
              } else {
                for (int i = 0; i < run; ++i) {
                  v64 a = aie::concat(aie::concat(aie::load_unaligned_v<16>(ap), aie::load_unaligned_v<16>(ap + rowb)),
                                      aie::concat(aie::load_unaligned_v<16>(ap + 2 * rowb), aie::load_unaligned_v<16>(ap + 3 * rowb)));
                  ap += cps;
                  c0.mac(a, aie::load_v<64>(w0)); w0 += wstride;
                  c1.mac(a, aie::load_v<64>(w1)); w1 += wstride;
                }
              }
            }
            remaining -= run;
            done += run;
            cp = 0;
            ++ti;
          }
          epi(og, t, c0);
          if (og + 1 < L.nb) epi(og + 1, t, c1);
        }
      }
      return;
    }
    // Gather: reduction step tt = tap_index * NCP + region, region fastest. The eight source
    // offsets/masks depend only on (tap, pixel tile) so they are rebuilt when the tap changes.
    const uint64_t flip64 = L.in_flip ? 0x8080808080808080ull : 0ull;
    int offs[8];
    uint64_t mask[8];
    for (int t = 0; t < L.t_out; ++t) {
      for (int og = 0; og < L.nb; og += 2) {
      const int og1 = og + 1 < L.nb ? og + 1 : og;
      MMUL c0, c1;
      if (L.first) {
        c0 = MMUL(bias_tile(L.bias + og * 8));
        c1 = MMUL(bias_tile(L.bias + og1 * 8));
      } else {
        c0 = MMUL(aie::load_v<64>(acc_buf + (og * L.t_out + t) * 64));
        c1 = MMUL(aie::load_v<64>(acc_buf + (og1 * L.t_out + t) * 64));
      }
      const int8_t *w0 = L.weights + (size_t)og * L.ttn * L.nbp * 64, *w1 = L.weights + (size_t)og1 * L.ttn * L.nbp * 64;
      int ti = L.ti0, cp = L.cp0, cur = -1;
      for (int tt = 0; tt < L.ttn; ++tt) {
        if (ti != cur) {
          cur = ti;
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
        gather_region(scratch_tiles, act + cp * L.reg, L.nbp, p_in, offs, mask, flip64);
        for (int l = 0; l < L.nbp; ++l) {
          v64 a = aie::load_v<64>(scratch_tiles + l * 64);
          c0.mac(a, aie::load_v<64>(w0)); w0 += 64;
          c1.mac(a, aie::load_v<64>(w1)); w1 += 64;
        }
        if (++cp == L.ncp) { cp = 0; ++ti; }
      }
      epi(og, t, c0);
      if (og + 1 < L.nb) epi(og + 1, t, c1);
      }
    }
  }
}
