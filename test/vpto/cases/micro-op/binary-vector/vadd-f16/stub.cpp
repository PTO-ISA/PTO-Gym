// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vadd-f16
// family: binary-vector
// target_ops: pto.vadd
// scenarios: core-f16, full-mask
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vadd_f16_kernel(__gm__ half *v1,
                                                  __gm__ half *v2,
                                                  __gm__ half *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
