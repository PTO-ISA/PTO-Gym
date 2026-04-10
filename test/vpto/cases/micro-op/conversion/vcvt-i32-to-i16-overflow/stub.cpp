#include <pto/common/type.hpp>
#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vcvt_i32_to_i16_overflow_kernel(
    __gm__ int32_t *v1, __gm__ int16_t *v2) {
  (void)v1;
  (void)v2;
}
