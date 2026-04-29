// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void online_softmax_update_kernel_2d(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6, __gm__ float *v7, int32_t v8, int32_t v9);

void LaunchSOFTMAX_f32_rows24_seq73(float *v1, float *v2, float *v3,
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
