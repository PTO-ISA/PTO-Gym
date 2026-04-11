// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pdintlv_b16-nontrivial
// family: materialization-predicate
// target_ops: pto.pdintlv_b16
// scenarios: predicate-transform, lane-order, nontrivial-pattern
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void pdintlv_b16_nontrivial_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
