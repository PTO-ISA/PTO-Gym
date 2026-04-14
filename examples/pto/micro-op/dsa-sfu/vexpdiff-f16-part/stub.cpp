// -----------------------------------------------------------------------------
// case: micro-op/dsa-sfu/vexpdiff-f16-part
// family: dsa-sfu
// target_ops: pto.vexpdiff
// scenarios: core-f16, fused-expdiff, part-even-odd
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vexpdiff_f16_part_kernel_2d(__gm__ half *v1,
                                                              __gm__ half *v2,
                                                              __gm__ float *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
