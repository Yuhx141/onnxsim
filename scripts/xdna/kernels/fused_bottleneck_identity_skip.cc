#include <stdint.h>
#ifndef FUSED_W
#define FUSED_W 1
#endif
#ifndef FUSED_C
#define FUSED_C 1
#endif
#ifndef FUSED_OUT_H
#define FUSED_OUT_H 1
#endif
#ifndef FUSED_OUT_W
#define FUSED_OUT_W FUSED_W
#endif
#ifndef FUSED_OUT_C
#define FUSED_OUT_C FUSED_C
#endif
#ifndef FUSED_SKIP_STRIDE
#define FUSED_SKIP_STRIDE 1
#endif

extern "C" void fused_bottleneck_identity_skip(const int8_t *input, int8_t *output) {
  const int pixels = FUSED_OUT_H * FUSED_OUT_W;
  for (int p = 0; p < pixels; ++p) {
    const int ip = (p / FUSED_OUT_W) * FUSED_SKIP_STRIDE * FUSED_W +
                   (p % FUSED_OUT_W) * FUSED_SKIP_STRIDE;
    for (int c = 0; c < FUSED_OUT_C; ++c)
      output[p * FUSED_OUT_C + c] = input[ip * FUSED_C + c];
  }
}
