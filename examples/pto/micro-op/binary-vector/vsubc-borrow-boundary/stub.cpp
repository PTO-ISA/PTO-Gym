// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vsubc-borrow-boundary
// family: binary-vector
// target_ops: pto.vsubc
// scenarios: core-u32-unsigned, full-mask, carry-chain
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void
vsubc_borrow_boundary_kernel_2d(__gm__ uint32_t *v1, __gm__ uint32_t *v2,
                                __gm__ uint32_t *v3, __gm__ uint8_t *v4) {
  (void)v1;
  (void)v2;
  (void)v3;
  (void)v4;
}
