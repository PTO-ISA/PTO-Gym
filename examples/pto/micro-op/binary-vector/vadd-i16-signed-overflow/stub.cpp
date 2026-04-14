#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vadd_i16_signed_overflow_kernel(__gm__ int16_t *v1,
                                                                  __gm__ int16_t *v2,
                                                                  __gm__ int16_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
