#include <stdint.h>

extern "C" void maxpool2d_nchw_f32(
    const float *input, float *output, int32_t input_width,
    int32_t output_width, int32_t kernel_height, int32_t kernel_width,
    int32_t stride_height, int32_t stride_width, int32_t output_rows) {
  for (int32_t oh = 0; oh < output_rows; ++oh) {
    for (int32_t ow = 0; ow < output_width; ++ow) {
      float maximum = -__builtin_inff();
      for (int32_t kh = 0; kh < kernel_height; ++kh) {
        for (int32_t kw = 0; kw < kernel_width; ++kw) {
          const int32_t row = oh * stride_height + kh;
          const int32_t col = ow * stride_width + kw;
          const float value = input[row * input_width + col];
          maximum = value > maximum ? value : maximum;
        }
      }
      output[oh * output_width + ow] = maximum;
    }
  }
}
