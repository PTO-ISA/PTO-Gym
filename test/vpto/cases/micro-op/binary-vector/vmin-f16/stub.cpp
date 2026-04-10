// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vmin-f16
// family: binary-vector
// target_ops: pto.vmin
// scenarios: core-f16, full-mask
// -----------------------------------------------------------------------------
#include <cstdint>
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vmin_f16_kernel(__gm__ half *v1,
                                                  __gm__ half *v2,
                                                  __gm__ half *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
