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

// Case 0: i32 32x64
extern "C" __global__ AICORE void TXORS_i32_32x64(__gm__ int32_t *src, __gm__ int32_t *dst, int32_t scalar);

void LaunchTXORS_i32_32x64(int32_t *src, int32_t *dst, void *stream) {
    TXORS_i32_32x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst, (int32_t)3);
}

// Case 1: i16 63x64
extern "C" __global__ AICORE void TXORS_i16_63x64(__gm__ int16_t *src, __gm__ int16_t *dst, int16_t scalar);

void LaunchTXORS_i16_63x64(int16_t *src, int16_t *dst, void *stream) {
    TXORS_i16_63x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst, (int16_t)3);
}

// Case 2: i32 31x128
extern "C" __global__ AICORE void TXORS_i32_31x128(__gm__ int32_t *src, __gm__ int32_t *dst, int32_t scalar);

void LaunchTXORS_i32_31x128(int32_t *src, int32_t *dst, void *stream) {
    TXORS_i32_31x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst, (int32_t)3);
}

// Case 3: i16 15x192
extern "C" __global__ AICORE void TXORS_i16_15x192(__gm__ int16_t *src, __gm__ int16_t *dst, int16_t scalar);

void LaunchTXORS_i16_15x192(int16_t *src, int16_t *dst, void *stream) {
    TXORS_i16_15x192<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst, (int16_t)3);
}
