// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vmin-bf16
// family: binary-vector
// target_ops: pto.vmin
// scenarios: core-bf16, full-mask
// -----------------------------------------------------------------------------
#include <cstdint>
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vmin_bf16_kernel(__gm__ bfloat16_t *v1,
                                                   __gm__ bfloat16_t *v2,
                                                   __gm__ bfloat16_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
