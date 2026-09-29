// NCHW uint8 max-pool kernel. Callers fill padded cells with zero; for a
// ReLU-quantized tensor with a positive zero point, zero is below every valid
// value and therefore matches negative-infinity padding before quantization.
#include <stdint.h>

extern "C" void maxpool2d_nchw_u8(
    const uint8_t *input, uint8_t *output,
    int32_t input_width, int32_t output_width,
    int32_t kernel_height, int32_t kernel_width,
    int32_t stride_height, int32_t stride_width,
    int32_t tile_output_rows, int32_t tile_channels) {
  const int input_rows = (tile_output_rows - 1) * stride_height + kernel_height;
  for (int c = 0; c < tile_channels; ++c) {
    for (int oy = 0; oy < tile_output_rows; ++oy) {
      for (int ox = 0; ox < output_width; ++ox) {
        uint8_t maximum = 0;
        for (int ky = 0; ky < kernel_height; ++ky) {
          const int iy = oy * stride_height + ky;
          for (int kx = 0; kx < kernel_width; ++kx) {
            const int ix = ox * stride_width + kx;
            const uint8_t value = input[c * input_rows * input_width + iy * input_width + ix];
            if (value > maximum) maximum = value;
          }
        }
        output[(oy * output_width + ox) * tile_channels + c] = maximum;
      }
    }
  }
}
