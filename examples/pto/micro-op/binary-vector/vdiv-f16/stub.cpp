// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vdiv-f16
// family: binary-vector
// target_ops: pto.vdiv
// scenarios: core-f16, full-mask
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vdiv_f16_kernel(__gm__ half *v1,
                                                  __gm__ half *v2,
                                                  __gm__ half *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
