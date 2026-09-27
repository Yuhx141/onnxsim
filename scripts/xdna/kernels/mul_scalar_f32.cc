#include <stdint.h>

#ifndef MUL_SCALAR
#define MUL_SCALAR 1.0f
#endif

extern "C" void mul_scalar_f32(const float *input, float *output, int32_t elements) {
  for (int32_t i = 0; i < elements; ++i)
    output[i] = input[i] * MUL_SCALAR;
}
