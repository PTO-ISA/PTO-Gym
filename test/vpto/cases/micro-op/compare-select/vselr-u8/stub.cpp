// -----------------------------------------------------------------------------
// case: micro-op/compare-select/vselr-u8
// family: compare-select
// target_ops: pto.vselr
// scenarios: core-u8, full-mask, explicit-lane-index
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>
#include <stdint.h>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vselr_u8_kernel_2d(__gm__ uint8_t *v1,
                                                     __gm__ uint8_t *v2,
                                                     __gm__ uint8_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
