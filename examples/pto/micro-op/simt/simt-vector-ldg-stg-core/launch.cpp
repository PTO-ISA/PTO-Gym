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
#include <stdint.h>
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif
extern "C" __global__ [aicore] void simt_vector_ldg_stg_core_kernel(
    __gm__ float *f32x2, __gm__ half *f16x2, __gm__ bfloat16_t *bf16x2,
    __gm__ int16_t *i16x2, __gm__ int32_t *i32x2,
    __gm__ void *hif8x2, __gm__ void *i8x2,
    __gm__ void *fp8e4x2, __gm__ void *fp8e5x2);
void LaunchSimt_vector_ldg_stg_core_kernel(
    void *f32x2, void *f16x2, void *bf16x2, void *i16x2, void *i32x2,
    void *hif8x2, void *i8x2, void *fp8e4x2, void *fp8e5x2, void *stream) {
  simt_vector_ldg_stg_core_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)f32x2, (__gm__ half *)f16x2, (__gm__ bfloat16_t *)bf16x2,
      (__gm__ int16_t *)i16x2, (__gm__ int32_t *)i32x2,
      (__gm__ void *)hif8x2, (__gm__ void *)i8x2,
      (__gm__ void *)fp8e4x2, (__gm__ void *)fp8e5x2);
}
