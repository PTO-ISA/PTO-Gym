// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Merged vadd launch wrappers: 9 variants (f32, f16, bf16, f32-exceptional,
// i16-signed, i16-signed-overflow, i16-unsigned, i16-unsigned-overflow, tail)
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

// --- f32 ---
extern "C" __global__ [aicore] void add_kernel_2d(__gm__ float *v1,
                                                 __gm__ float *v2,
                                                 __gm__ float *v3);

extern "C" __global__ [aicore] void vadd_deep_merged_kernel(
    __gm__ float * arg0,
    __gm__ float * arg1,
    __gm__ float * arg2,
    __gm__ half * arg3,
    __gm__ half * arg4,
    __gm__ half * arg5,
    __gm__ bfloat16_t * arg6,
    __gm__ bfloat16_t * arg7,
    __gm__ bfloat16_t * arg8,
    __gm__ float * arg9,
    __gm__ float * arg10,
    __gm__ float * arg11,
    __gm__ int16_t * arg12,
    __gm__ int16_t * arg13,
    __gm__ int16_t * arg14,
    __gm__ int16_t * arg15,
    __gm__ int16_t * arg16,
    __gm__ int16_t * arg17,
    __gm__ uint16_t * arg18,
    __gm__ uint16_t * arg19,
    __gm__ uint16_t * arg20,
    __gm__ uint16_t * arg21,
    __gm__ uint16_t * arg22,
    __gm__ uint16_t * arg23,
    __gm__ float * arg24,
    __gm__ float * arg25,
    __gm__ float * arg26);

void LaunchVaddDeepMerged(float * p0, float * p1, float * p2, uint16_t * p3, uint16_t * p4, uint16_t * p5, uint16_t * p6, uint16_t * p7, uint16_t * p8, float * p9, float * p10, float * p11, int16_t * p12, int16_t * p13, int16_t * p14, int16_t * p15, int16_t * p16, int16_t * p17, uint16_t * p18, uint16_t * p19, uint16_t * p20, uint16_t * p21, uint16_t * p22, uint16_t * p23, float * p24, float * p25, float * p26, void *stream) {
  vadd_deep_merged_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)p0,
      (__gm__ float *)p1,
      (__gm__ float *)p2,
      (__gm__ half *)p3,
      (__gm__ half *)p4,
      (__gm__ half *)p5,
      (__gm__ bfloat16_t *)p6,
      (__gm__ bfloat16_t *)p7,
      (__gm__ bfloat16_t *)p8,
      (__gm__ float *)p9,
      (__gm__ float *)p10,
      (__gm__ float *)p11,
      (__gm__ int16_t *)p12,
      (__gm__ int16_t *)p13,
      (__gm__ int16_t *)p14,
      (__gm__ int16_t *)p15,
      (__gm__ int16_t *)p16,
      (__gm__ int16_t *)p17,
      (__gm__ uint16_t *)p18,
      (__gm__ uint16_t *)p19,
      (__gm__ uint16_t *)p20,
      (__gm__ uint16_t *)p21,
      (__gm__ uint16_t *)p22,
      (__gm__ uint16_t *)p23,
      (__gm__ float *)p24,
      (__gm__ float *)p25,
      (__gm__ float *)p26);
}
