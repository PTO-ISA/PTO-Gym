// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Merged launch wrappers
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
struct MrgSortExecutedNumList { uint16_t mrgSortList0,mrgSortList1,mrgSortList2,mrgSortList3; };
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void vshl_i16_unsigned_kernel(__gm__ uint16_t *v1,
                                                           __gm__ uint16_t *v2,
                                                           __gm__ uint16_t *v3);

extern "C" __global__ [aicore] void vshl_deep_merged_kernel(
    __gm__ uint16_t * arg0,
    __gm__ uint16_t * arg1,
    __gm__ uint16_t * arg2,
    __gm__ uint32_t * arg3,
    __gm__ uint32_t * arg4,
    __gm__ uint32_t * arg5,
    __gm__ uint16_t * arg6,
    __gm__ uint16_t * arg7,
    __gm__ uint16_t * arg8);

void LaunchVshlDeepMerged(uint16_t * p0, uint16_t * p1, uint16_t * p2, uint32_t * p3, uint32_t * p4, uint32_t * p5, uint16_t * p6, uint16_t * p7, uint16_t * p8, void *stream) {
  vshl_deep_merged_kernel<<<1, nullptr, stream>>>(
      (__gm__ uint16_t *)p0,
      (__gm__ uint16_t *)p1,
      (__gm__ uint16_t *)p2,
      (__gm__ uint32_t *)p3,
      (__gm__ uint32_t *)p4,
      (__gm__ uint32_t *)p5,
      (__gm__ uint16_t *)p6,
      (__gm__ uint16_t *)p7,
      (__gm__ uint16_t *)p8);
}
