// -----------------------------------------------------------------------------
// case: micro-op/dsa-sfu/vmula
// family: dsa-sfu
// target_ops: pto.vmula
// scenarios: core-f32, fused-op, accumulator
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vec_add_scalar_kernel_2d(__gm__ float *v1,
                                                                __gm__ float *v2) {
  (void)v1;
  (void)v2;
}
