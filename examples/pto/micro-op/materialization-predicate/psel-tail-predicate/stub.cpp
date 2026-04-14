// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/psel-tail-predicate
// family: materialization-predicate
// target_ops: pto.psel
// scenarios: predicate-transform, predicate-select, tail-mask

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void psel_tail_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
