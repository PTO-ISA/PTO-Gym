// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/flash-attention
// family: kernels
// target_ops: pto.mte_gm_l1_frac, pto.mte_l1_l0a, pto.mte_l1_l0b, pto.mad,
//   pto.mte_l0c_ub, pto.mte_gm_ub, pto.mte_ub_gm, pto.vlds, pto.vcmax,
//   pto.vdup, pto.vmax, pto.vexpdif, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv,
//   pto.vsts, pto.sync.set, pto.sync.wait
// scenarios: flash-attention, cube-qk, tiled-online-softmax, q32-k32-d8
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
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void flash_attention_kernel_2d(
    __gm__ float *q, __gm__ float *k, __gm__ float *value_t, __gm__ float *out,
    int32_t seq, int32_t rows);

void LaunchFlash_attention_kernel_2d(float *q, float *k, float *value_t,
                                     float *out, int32_t seq, int32_t rows,
                                     void *stream) {
  const int32_t blockRows = 16;
  const int32_t blocks = (rows + blockRows - 1) / blockRows;
  flash_attention_kernel_2d<<<blocks, nullptr, stream>>>(
      (__gm__ float *)q, (__gm__ float *)k, (__gm__ float *)value_t,
      (__gm__ float *)out, seq, rows);
}
