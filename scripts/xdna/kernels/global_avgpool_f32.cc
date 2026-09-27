#include <stdint.h>

extern "C" void global_avgpool_f32(const float *input, float *output,
                                    int32_t spatial, int32_t channels) {
  for (int32_t c = 0; c < channels; ++c) {
    float sum = 0.0f;
    const float *channel = input + c * spatial;
    for (int32_t p = 0; p < spatial; ++p)
      sum += channel[p];
    output[c] = sum / (float)spatial;
  }
}
