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

// Case 0: f32 16x64
extern "C" __global__ AICORE void TSQRT_f32_16x64(__gm__ float *a, __gm__ float *b);

void LaunchTSQRT_f32_16x64(void *a, void *b, void *stream) {
    TSQRT_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b);
}

// Case 1: f32 32x32
extern "C" __global__ AICORE void TSQRT_f32_32x32(__gm__ float *a, __gm__ float *b);

void LaunchTSQRT_f32_32x32(void *a, void *b, void *stream) {
    TSQRT_f32_32x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b);
}

// Case 2: f16 16x64
extern "C" __global__ AICORE void TSQRT_f16_16x64(__gm__ uint16_t *a, __gm__ uint16_t *b);

void LaunchTSQRT_f16_16x64(void *a, void *b, void *stream) {
    TSQRT_f16_16x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b);
}

// Case 3: f16 32x32
extern "C" __global__ AICORE void TSQRT_f16_32x32(__gm__ uint16_t *a, __gm__ uint16_t *b);

void LaunchTSQRT_f16_32x32(void *a, void *b, void *stream) {
    TSQRT_f16_32x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b);
}