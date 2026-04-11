#include <pto/common/type.hpp>
#include <stdint.h>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vadds_bf16_kernel(__gm__ bfloat16_t *v1,
                                                    __gm__ bfloat16_t *v2) {
  (void)v1;
  (void)v2;
}
