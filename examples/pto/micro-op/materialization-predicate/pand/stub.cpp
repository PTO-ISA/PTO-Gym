// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pand
// family: materialization-predicate
// target_ops: pto.pand
// scenarios: predicate-transform
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void pand_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
