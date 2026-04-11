// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vmin-i16-signed
// family: binary-vector
// target_ops: pto.vmin
// scenarios: core-i16-signed, full-mask
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

extern "C" __global__ AICORE void vmin_i16_signed_kernel(__gm__ int16_t *v1,
                                                         __gm__ int16_t *v2,
                                                         __gm__ int16_t *v3);

void LaunchVmin_i16_signed_kernel(int16_t *v1, int16_t *v2, int16_t *v3,
                                  void *stream) {
  vmin_i16_signed_kernel<<<1, nullptr, stream>>>((__gm__ int16_t *)v1,
                                                 (__gm__ int16_t *)v2,
                                                 (__gm__ int16_t *)v3);
}
