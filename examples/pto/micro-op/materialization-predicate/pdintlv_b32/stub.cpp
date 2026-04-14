// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pdintlv_b32
// family: materialization-predicate
// target_ops: pto.pdintlv_b32
// scenarios: predicate-transform, lane-order

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void pdintlv_b32_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
