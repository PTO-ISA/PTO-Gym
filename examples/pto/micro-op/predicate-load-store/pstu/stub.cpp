// -----------------------------------------------------------------------------
// case: micro-op/predicate-load-store/pstu
// family: predicate-load-store
// target_ops: pto.pstu
// scenarios: unaligned-predicate-store, state-update, representative-logical-elements
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void pstu_kernel_2d(__gm__ float *v1,
                                                 __gm__ float *v2,
                                                 __gm__ uint32_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
