// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pintlv_b16
// family: materialization-predicate
// target_ops: pto.pintlv_b16
// scenarios: predicate-transform, lane-order

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void pintlv_b16_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
