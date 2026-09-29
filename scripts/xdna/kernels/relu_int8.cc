#include <stdint.h>

extern "C" void relu_int8_kernel(const int8_t *input, int8_t *output, int32_t unused, int32_t elements) {
  (void)unused;
  for (int32_t i = 0; i < elements; ++i) {
    const int8_t value = input[i];
    output[i] = value < 0 ? 0 : value;
  }
}
