// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/plt-tail-mask-boundary
// family: materialization-predicate
// target_ops: pto.plt_b16, pto.plt_b32, pto.plt_b8
// scenarios: tail-mask, scalar-carry-out, boundary
// ---------------------------------------------------------------------------
// PTOAS compatibility layer
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

extern "C" __global__ AICORE void plt_tail_mask_boundary_kernel_2d(__gm__ uint32_t *v1);

void LaunchPltTailMaskBoundary(uint32_t *v1, void *stream) {
  plt_tail_mask_boundary_kernel_2d<<<1, nullptr, stream>>>((__gm__ uint32_t *)v1);
}
