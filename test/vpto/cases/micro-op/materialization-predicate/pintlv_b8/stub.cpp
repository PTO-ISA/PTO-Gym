// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pintlv_b8
// family: materialization-predicate
// target_ops: pto.pintlv_b8
// scenarios: predicate-transform, lane-order
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void pintlv_b8_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
