// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pxor
// family: materialization-predicate
// target_ops: pto.pxor
// scenarios: predicate-transform

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void pxor_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
