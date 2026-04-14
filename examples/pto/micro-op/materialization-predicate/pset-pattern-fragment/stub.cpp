// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pset-pattern-fragment
// family: materialization-predicate
// target_ops: pto.pset_b16, pto.pset_b32, pto.pset_b8
// scenarios: pattern-mask, fragment-pattern

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void pset_pattern_fragment_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
