// -----------------------------------------------------------------------------
// case: micro-op/vec-scalar/vaddcs-carry-boundary
// family: vec-scalar
// target_ops: pto.vaddcs
// scenarios: core-u32-unsigned, full-mask, carry-chain, integer-overflow
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

extern "C" __global__ AICORE void vaddcs_carry_boundary_kernel(
    __gm__ uint32_t *v1, __gm__ uint32_t *v2, __gm__ uint32_t *v3,
    __gm__ uint8_t *v4);

void LaunchVaddcsCarryBoundaryKernel(uint32_t *v1, uint32_t *v2, uint32_t *v3,
                                     uint8_t *v4, void *stream) {
  vaddcs_carry_boundary_kernel<<<1, nullptr, stream>>>(
      (__gm__ uint32_t *)v1, (__gm__ uint32_t *)v2, (__gm__ uint32_t *)v3,
      (__gm__ uint8_t *)v4);
}
