// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/vdup-lane
// family: materialization-predicate
// target_ops: pto.vdup
// scenarios: core-f32, vector-input, lowest-highest
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vdup_lane_kernel_2d(__gm__ float *src,
                                                      __gm__ float *outLow,
                                                      __gm__ float *outHigh) {
  (void)src;
  (void)outLow;
  (void)outHigh;
}
