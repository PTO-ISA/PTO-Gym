// -----------------------------------------------------------------------------
// case: micro-op/dsa-sfu/vmull
// family: dsa-sfu
// target_ops: pto.vmull
// scenarios: widening-op, hi-lo-split
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vmull_kernel_2d(__gm__ int *v1,
                                                  __gm__ int *v2) {
  (void)v1;
  (void)v2;
}
