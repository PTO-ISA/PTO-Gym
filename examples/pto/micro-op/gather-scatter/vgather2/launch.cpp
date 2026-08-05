// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
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

extern "C" __global__ [aicore] void vgather2_deep_merged_kernel(
    __gm__ float * arg0,
    __gm__ int32_t * arg1,
    __gm__ float * arg2,
    __gm__ float * arg3,
    __gm__ int32_t * arg4,
    __gm__ float * arg5,
    __gm__ half * arg6,
    __gm__ uint16_t * arg7,
    __gm__ half * arg8,
    __gm__ uint8_t * arg9,
    __gm__ uint16_t * arg10,
    __gm__ int8_t * arg11,
    __gm__ int16_t * arg12);

void LaunchVgather2DeepMerged(float * p0, int * p1, float * p2,
                              uint16_t * p3, uint16_t * p4, uint16_t * p5,
                              uint8_t * p6, uint16_t * p7,
                              int8_t * p8, int16_t * p9,
                              void *stream) {
  vgather2_deep_merged_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)p0,
      (__gm__ int32_t *)p0,
      (__gm__ float *)p0,
      (__gm__ float *)p0,
      (__gm__ int32_t *)p1,
      (__gm__ float *)p2,
      (__gm__ half *)p3,
      (__gm__ uint16_t *)p4,
      (__gm__ half *)p5,
      (__gm__ uint8_t *)p6,
      (__gm__ uint16_t *)p7,
      (__gm__ int8_t *)p8,
      (__gm__ int16_t *)p9);
}
