// Row kernels for a streamed, fused INT8 identity ResNet bottleneck.
// Weight and bias tiles are stage-local so the three stages fit separate AIE
// cores instead of requiring the full block constants in one core's memory.

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
#ifndef FUSED_RESIDUAL_SHIFT
#error "FUSED_RESIDUAL_SHIFT must be defined"
#endif
#ifndef FUSED_INPUT_SHIFT
#error "FUSED_INPUT_SHIFT must be defined"
#endif

static int32_t round_shift_even(int32_t value, int shift) {
  if (shift <= 0)
    return value;
  int64_t wide = value;
  uint64_t magnitude = wide < 0 ? (uint64_t)(-wide) : (uint64_t)wide;
  uint64_t quotient = magnitude >> shift;
  uint64_t remainder = magnitude & ((((uint64_t)1) << shift) - 1);
  uint64_t halfway = ((uint64_t)1) << (shift - 1);
  if (remainder > halfway || (remainder == halfway && (quotient & 1)))
    ++quotient;
  return wide < 0 ? -(int32_t)quotient : (int32_t)quotient;
}

static int64_t round_shift_even64(int64_t value, int shift) {
  if (shift <= 0)
    return value;
  uint64_t magnitude = value < 0 ? (uint64_t)(-value) : (uint64_t)value;
  uint64_t quotient = magnitude >> shift;
  uint64_t remainder = magnitude & ((((uint64_t)1) << shift) - 1);
  uint64_t halfway = ((uint64_t)1) << (shift - 1);
  if (remainder > halfway || (remainder == halfway && (quotient & 1)))
    ++quotient;
  return value < 0 ? -(int64_t)quotient : (int64_t)quotient;
}

static int64_t scale_pow2(int32_t value, int exponent, int common_shift) {
  return ((int64_t)value) * (((int64_t)1) << (exponent + common_shift));
}

extern "C" void fused_bottleneck_conv1_row(const int8_t *input,
                                            const uint8_t *params,
                                            uint8_t *output) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS1_OFFSET);
  for (int x = 0; x < FUSED_W; ++x) {
    for (int oc = 0; oc < FUSED_MID; ++oc) {
      int32_t acc = bias[oc];
      for (int ic = 0; ic < FUSED_C; ++ic)
        acc += (int32_t)input[x * FUSED_C + ic] *
               (int32_t)weights[oc * FUSED_C + ic];
      int32_t q = 128 + round_shift_even(acc, FUSED_SHIFT1);
      output[x * FUSED_MID + oc] = (uint8_t)(q < 128 ? 128 : (q > 255 ? 255 : q));
    }
  }
}

extern "C" void fused_bottleneck_conv2_row(const uint8_t *top,
                                            const uint8_t *middle,
                                            const uint8_t *bottom,
                                            const uint8_t *params,
                                            uint8_t *output,
                                            int32_t row,
                                            int32_t channel_offset) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS2_OFFSET);
  for (int x = 0; x < FUSED_W; ++x) {
    for (int oc = 0; oc < FUSED_MID / 2; ++oc) {
      int full_oc = channel_offset + oc;
      int32_t acc = bias[full_oc];
      for (int ky = 0; ky < 3; ++ky) {
        if ((ky == 0 && row == 0) || (ky == 2 && row == FUSED_H - 1))
          continue;
        const uint8_t *source = ky == 0 ? top : (ky == 1 ? middle : bottom);
        for (int kx = 0; kx < 3; ++kx) {
          int ix = x + kx - 1;
          if (ix < 0 || ix >= FUSED_W)
            continue;
          for (int ic = 0; ic < FUSED_MID; ++ic) {
            int weight_index = ((full_oc * FUSED_MID + ic) * 3 + ky) * 3 + kx;
            acc += ((int32_t)source[ix * FUSED_MID + ic] - 128) *
                   (int32_t)weights[weight_index];
          }
        }
      }
      int32_t q = 128 + round_shift_even(acc, FUSED_SHIFT2);
      output[x * (FUSED_MID / 2) + oc] = (uint8_t)(q < 128 ? 128 : (q > 255 ? 255 : q));
    }
  }
}

extern "C" void fused_bottleneck_conv3_residual_row(const uint8_t *main_input0,
                                                     const uint8_t *main_input1,
                                                     const uint8_t *params,
                                                     const int8_t *residual,
                                                     uint8_t *output) {
  const int8_t *weights = (const int8_t *)params;
  const int32_t *bias = (const int32_t *)(params + FUSED_BIAS3_OFFSET);
  for (int x = 0; x < FUSED_W; ++x) {
    for (int oc = 0; oc < FUSED_C; ++oc) {
      int32_t acc = bias[oc];
      for (int ic = 0; ic < FUSED_MID / 2; ++ic) {
        acc += ((int32_t)main_input0[x * (FUSED_MID / 2) + ic] - 128) *
               (int32_t)weights[oc * FUSED_MID + ic];
        acc += ((int32_t)main_input1[x * (FUSED_MID / 2) + ic] - 128) *
               (int32_t)weights[oc * FUSED_MID + ic + FUSED_MID / 2];
      }
      int32_t q3 = 128 + round_shift_even(acc, FUSED_SHIFT3);
      if (q3 < 0)
        q3 = 0;
      else if (q3 > 255)
        q3 = 255;
      const int common_shift = FUSED_RESIDUAL_SHIFT < FUSED_INPUT_SHIFT
                                   ? (FUSED_RESIDUAL_SHIFT < 0 ? -FUSED_RESIDUAL_SHIFT : 0)
                                   : (FUSED_INPUT_SHIFT < 0 ? -FUSED_INPUT_SHIFT : 0);
      int64_t sum = scale_pow2(q3 - 128, FUSED_RESIDUAL_SHIFT, common_shift) +
                    scale_pow2(residual[x * FUSED_C + oc], FUSED_INPUT_SHIFT, common_shift);
      int64_t value = round_shift_even64(sum, common_shift);
      if (value < 0)
        value = 0;
      else if (value > 127)
        value = 127;
      output[x * FUSED_C + oc] = (uint8_t)(128 + value);
    }
  }
}
