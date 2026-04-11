// -----------------------------------------------------------------------------
// case: micro-op/vec-scalar/vshrs-shift-boundary
// family: vec-scalar
// target_ops: pto.vshrs
// scenarios: core-i16-unsigned, full-mask, scalar-operand
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

extern "C" __global__ AICORE void vshrs_shift_boundary_kernel(__gm__ float *v1,
                                                              __gm__ float *v2) {
  (void)v1;
  (void)v2;
}
