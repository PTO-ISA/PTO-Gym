// -----------------------------------------------------------------------------
// case: micro-op/vector-load-store/vlds-unpk-b16
// family: vector-load-store
// target_ops: pto.vlds
// scenarios: core-f16, full-mask, aligned, dist-unpk-b16
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vlds_unpk_b16_kernel_2d(__gm__ half *v1,
                                                          __gm__ half *v2) {
  (void)v1;
  (void)v2;
}
