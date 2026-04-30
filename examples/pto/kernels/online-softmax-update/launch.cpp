// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/online-softmax-update
// family: kernels
// target_ops: pto.get_block_idx, pto.dma_load, pto.dma_store, pto.vlds, pto.vcmax, pto.vdup, pto.vmax, pto.vexpdif, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv, pto.vsts
// scenarios: online-softmax-update, dynamic-rows-and-seq, max-seq-128, block-rows-8, oldmax-oldsum-qk-to-newmax-newsum-expmax-out
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

extern "C" __global__ AICORE void online_softmax_update_kernel_2d(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3,
    __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, int32_t v8, int32_t v9);

void LaunchOnline_softmax_update_kernel_2d(float *v1, float *v2, float *v3,
                                           float *v4, float *v5, float *v6,
                                           float *v7, int32_t v8, int32_t v9,
                                           void *stream) {
  const int32_t blockRows = 8;
  const int32_t blocks = (v9 + blockRows - 1) / blockRows;
  online_softmax_update_kernel_2d<<<blocks, nullptr, stream>>>(
      (__gm__ float *)v1, (__gm__ float *)v2, (__gm__ float *)v3,
      (__gm__ float *)v4, (__gm__ float *)v5, (__gm__ float *)v6,
      (__gm__ float *)v7, v8, v9);
}
