#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vtrc_f16_rounding_kernel_2d(__gm__ half *v1,
                                                              __gm__ half *v2) {
  (void)v1;
  (void)v2;
}
