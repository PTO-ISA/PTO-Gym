// -----------------------------------------------------------------------------
// case: micro-op/compare-select/vselr
// family: compare-select
// target_ops: pto.vselr
// scenarios: core-f32, full-mask, explicit-lane-index
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vselr_kernel_2d(__gm__ float *v1,
                                                  __gm__ int *v2,
                                                  __gm__ float *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
