// -----------------------------------------------------------------------------
// case: micro-op/dsa-sfu/vaxpy-f32
// family: dsa-sfu
// target_ops: pto.vaxpy
// scenarios: core-f32, scalar-operand, fused-op
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vaxpy_kernel_2d(__gm__ float *v1,
                                                  __gm__ float *v2,
                                                  __gm__ float *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
