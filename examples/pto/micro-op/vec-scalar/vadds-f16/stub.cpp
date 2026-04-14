#include <stdint.h>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vadds_f16_kernel(__gm__ half *v1,
                                                   __gm__ half *v2) {
  (void)v1;
  (void)v2;
}
