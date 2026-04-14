// ---------------------------------------------------------------------------
// PTOAS compatibility layer
// ---------------------------------------------------------------------------
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

#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void vbitsort_kernel_f32(__gm__ float *scores,
                                                      __gm__ uint32_t *indices,
                                                      __gm__ uint32_t *output);

void LaunchVbitsort_kernel_f32(float *scores, uint32_t *indices, uint32_t *output,
                               void *stream) {
  vbitsort_kernel_f32<<<1, nullptr, stream>>>((__gm__ float *)scores,
                                              (__gm__ uint32_t *)indices,
                                              (__gm__ uint32_t *)output);
}
