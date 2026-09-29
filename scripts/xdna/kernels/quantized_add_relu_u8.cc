#include <stdint.h>

#ifndef ADD_MULT_A
#define ADD_MULT_A 1073741824LL
#endif
#ifndef ADD_MULT_B
#define ADD_MULT_B 1073741824LL
#endif
#ifndef ADD_ZERO_A
#define ADD_ZERO_A 0
#endif
#ifndef ADD_ZERO_B
#define ADD_ZERO_B 0
#endif
#ifndef ADD_ZERO_OUT
#define ADD_ZERO_OUT 0
#endif

extern "C" void quantized_add_relu_u8(const uint8_t *a, const uint8_t *b,
                                       uint8_t *out, int32_t elements) {
  for (int32_t i = 0; i < elements; ++i) {
    const int64_t accum = ((int32_t)a[i] - ADD_ZERO_A) * ADD_MULT_A +
                          ((int32_t)b[i] - ADD_ZERO_B) * ADD_MULT_B;
    if (accum <= 0) {
      out[i] = ADD_ZERO_OUT;
      continue;
    }
    const uint64_t magnitude = (uint64_t)accum;
    uint64_t rounded = magnitude >> 30;
    const uint64_t remainder = magnitude & ((1ULL << 30) - 1);
    const uint64_t halfway = 1ULL << 29;
    if (remainder > halfway || (remainder == halfway && (rounded & 1)))
      ++rounded;
    const uint64_t quantized = rounded + ADD_ZERO_OUT;
    out[i] = quantized > 255 ? 255 : (uint8_t)quantized;
  }
}
