// -----------------------------------------------------------------------------
// case: micro-op/vec-scalar/vsubcs
// family: vec-scalar
// target_ops: pto.vsubcs
// scenarios: core-u32-unsigned, full-mask, carry-chain
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void
vsubcs_kernel(__gm__ uint32_t *v1, __gm__ uint32_t *v2, __gm__ uint32_t *v3,
              __gm__ uint8_t *v4) {
  (void)v1;
  (void)v2;
  (void)v3;
  (void)v4;
}
