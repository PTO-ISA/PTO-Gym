// -----------------------------------------------------------------------------
// case: micro-op/rearrangement/vpack-higher
// family: rearrangement
// target_ops: pto.vpack
// scenarios: narrowing, higher-half-placement, zero-fill-lower-half
// -----------------------------------------------------------------------------
#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vpack_higher_kernel_2d(__gm__ int *v1,
                                                           __gm__ uint16_t *v2) {
  (void)v1;
  (void)v2;
}
