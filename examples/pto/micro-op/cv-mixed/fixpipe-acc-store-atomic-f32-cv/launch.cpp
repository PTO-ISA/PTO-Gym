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

#include <cstdint>

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

extern "C" __global__ [aicore] void fixpipe_acc_store_atomic_f32_cv_kernel(
    __gm__ __fp16 *src, __gm__ __fp16 *id, __gm__ float *out_plain,
    __gm__ float *out_atomic_add, __gm__ float *out_atomic_max,
    __gm__ float *out_atomic_min);

void LaunchFixpipe_acc_store_atomic_f32_cv_kernel(__fp16 *src, __fp16 *id,
                                                  float *outPlain,
                                                  float *outAtomicAdd,
                                                  float *outAtomicMax,
                                                  float *outAtomicMin,
                                                  void *stream) {
  fixpipe_acc_store_atomic_f32_cv_kernel<<<1, nullptr, stream>>>(
      (__gm__ __fp16 *)src, (__gm__ __fp16 *)id, (__gm__ float *)outPlain,
      (__gm__ float *)outAtomicAdd, (__gm__ float *)outAtomicMax,
      (__gm__ float *)outAtomicMin);
}
