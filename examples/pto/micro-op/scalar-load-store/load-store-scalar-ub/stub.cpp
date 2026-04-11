#include <cstdint>
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void load_store_scalar_ub_kernel(__gm__ int16_t *v1,
                                                              __gm__ int16_t *v2) {
  (void)v1;
  (void)v2;
}
