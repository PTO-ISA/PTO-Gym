// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ... license ...
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

extern "C" __global__ [aicore] void min_kernel_2d(__gm__ float *v1,
                                                 __gm__ float *v2,
                                                 __gm__ float *v3);

extern "C" __global__ [aicore] void vmin_deep_merged_kernel(
    __gm__ float * arg0,
    __gm__ float * arg1,
    __gm__ float * arg2,
    __gm__ float * arg3,
    __gm__ float * arg4,
    __gm__ float * arg5,
    __gm__ half * arg6,
    __gm__ half * arg7,
    __gm__ half * arg8,
    __gm__ bfloat16_t * arg9,
    __gm__ bfloat16_t * arg10,
    __gm__ bfloat16_t * arg11,
    __gm__ int16_t * arg12,
    __gm__ int16_t * arg13,
    __gm__ int16_t * arg14,
    __gm__ uint16_t * arg15,
    __gm__ uint16_t * arg16,
    __gm__ uint16_t * arg17,
    __gm__ float * arg18,
    __gm__ float * arg19,
    __gm__ float * arg20);

void LaunchVminDeepMerged(float * p0, float * p1, float * p2, float * p3, float * p4, float * p5, uint16_t * p6, uint16_t * p7, uint16_t * p8, uint16_t * p9, uint16_t * p10, uint16_t * p11, int16_t * p12, int16_t * p13, int16_t * p14, uint16_t * p15, uint16_t * p16, uint16_t * p17, float * p18, float * p19, float * p20, void *stream) {
  vmin_deep_merged_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)p0,
      (__gm__ float *)p1,
      (__gm__ float *)p2,
      (__gm__ float *)p3,
      (__gm__ float *)p4,
      (__gm__ float *)p5,
      (__gm__ half *)p6,
      (__gm__ half *)p7,
      (__gm__ half *)p8,
      (__gm__ bfloat16_t *)p9,
      (__gm__ bfloat16_t *)p10,
      (__gm__ bfloat16_t *)p11,
      (__gm__ int16_t *)p12,
      (__gm__ int16_t *)p13,
      (__gm__ int16_t *)p14,
      (__gm__ uint16_t *)p15,
      (__gm__ uint16_t *)p16,
      (__gm__ uint16_t *)p17,
      (__gm__ float *)p18,
      (__gm__ float *)p19,
      (__gm__ float *)p20);
}
