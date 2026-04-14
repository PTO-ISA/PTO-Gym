#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vabs_i16_signed_overflow_edge_kernel(
    __gm__ int16_t *v1, __gm__ int16_t *v2) {
  (void)v1;
  (void)v2;
}
