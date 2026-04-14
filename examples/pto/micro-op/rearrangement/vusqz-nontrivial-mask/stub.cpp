// -----------------------------------------------------------------------------
// case: micro-op/rearrangement/vusqz-nontrivial-mask
// family: rearrangement
// target_ops: pto.vusqz
// scenarios: predicate-driven-rearrangement, prefix-count
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vusqz_nontrivial_mask_kernel_2d(__gm__ int32_t *v1,
                                                                  __gm__ float *v2,
                                                                  __gm__ int32_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
