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

// Case 0: f32 32x64 -> dst 32x128 (valid 32x64)
extern "C" __global__ AICORE void TLRELU_f32_32x64_dst128(__gm__ float *src, __gm__ float *dst, float slope);

void LaunchTLRELU_f32_32x64_dst128(float *src, float *dst, float slope, void *stream) {
    TLRELU_f32_32x64_dst128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, slope);
}

// Case 1: f16 63x64 -> dst 63x128 (valid 63x64)
extern "C" __global__ AICORE void TLRELU_f16_63x64_dst128(__gm__ uint16_t *src, __gm__ uint16_t *dst, float slope);

void LaunchTLRELU_f16_63x64_dst128(uint16_t *src, uint16_t *dst, float slope, void *stream) {
    TLRELU_f16_63x64_dst128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, slope);
}

// Case 2: f32 7x448 -> dst 7x512 (valid 7x448)
extern "C" __global__ AICORE void TLRELU_f32_7x448_dst512(__gm__ float *src, __gm__ float *dst, float slope);

void LaunchTLRELU_f32_7x448_dst512(float *src, float *dst, float slope, void *stream) {
    TLRELU_f32_7x448_dst512<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, slope);
}

// Case 3: f32 256x16 -> dst 256x32 (valid 256x16)
extern "C" __global__ AICORE void TLRELU_f32_256x16_dst32(__gm__ float *src, __gm__ float *dst, float slope);

void LaunchTLRELU_f32_256x16_dst32(float *src, float *dst, float slope, void *stream) {
    TLRELU_f32_256x16_dst32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, slope);
}