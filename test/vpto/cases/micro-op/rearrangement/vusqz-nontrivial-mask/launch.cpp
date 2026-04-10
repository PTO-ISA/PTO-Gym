// -----------------------------------------------------------------------------
// case: micro-op/rearrangement/vusqz-nontrivial-mask
// family: rearrangement
// target_ops: pto.vusqz
// scenarios: predicate-driven-rearrangement, prefix-count
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

#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ AICORE void vusqz_nontrivial_mask_kernel_2d(__gm__ int32_t *v1,
                                                                  __gm__ float *v2,
                                                                  __gm__ int32_t *v3);

void LaunchVusqz_nontrivial_mask_kernel_2d(int32_t *v1,
                                           float *v2,
                                           int32_t *v3,
                                           void *stream) {
  vusqz_nontrivial_mask_kernel_2d<<<1, nullptr, stream>>>(
      (__gm__ int32_t *)v1, (__gm__ float *)v2, (__gm__ int32_t *)v3);
}
