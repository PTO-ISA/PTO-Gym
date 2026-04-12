// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/vdup-lane
// family: materialization-predicate
// target_ops: pto.vdup
// scenarios: core-f32, vector-input, lowest-highest
// -----------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>

#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ AICORE void vdup_lane_kernel_2d(__gm__ float *src,
                                                      __gm__ float *outLow,
                                                      __gm__ float *outHigh);

void LaunchVdup_lane_kernel_2d(float *src, float *outLow, float *outHigh,
                               void *stream) {
  vdup_lane_kernel_2d<<<1, nullptr, stream>>>((__gm__ float *)src,
                                              (__gm__ float *)outLow,
                                              (__gm__ float *)outHigh);
}
