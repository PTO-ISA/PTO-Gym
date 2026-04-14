// -----------------------------------------------------------------------------
// case: micro-op/compare-select/vselr-f16
// family: compare-select
// target_ops: pto.vselr
// scenarios: core-f16, full-mask, explicit-lane-index
// -----------------------------------------------------------------------------
#include <stdint.h>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vselr_f16_kernel_2d(__gm__ half *v1,
                                                      __gm__ uint16_t *v2,
                                                      __gm__ half *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
