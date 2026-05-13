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

// Case 0: f16 64x64
extern "C" __global__ AICORE void TPRELU_f16_64x64(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTPRELU_f16_64x64(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TPRELU_f16_64x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 1: f16 63x63
extern "C" __global__ AICORE void TPRELU_f16_63x63(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTPRELU_f16_63x63(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TPRELU_f16_63x63<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 2: f16 1x16384
extern "C" __global__ AICORE void TPRELU_f16_1x16384(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTPRELU_f16_1x16384(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TPRELU_f16_1x16384<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 3: f16 2048x16
extern "C" __global__ AICORE void TPRELU_f16_2048x16(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTPRELU_f16_2048x16(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TPRELU_f16_2048x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 4: f32 64x64
extern "C" __global__ AICORE void TPRELU_f32_64x64(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTPRELU_f32_64x64(float *src0, float *src1, float *dst, void *stream) {
    TPRELU_f32_64x64<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 5: f32 63x63
extern "C" __global__ AICORE void TPRELU_f32_63x63(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTPRELU_f32_63x63(float *src0, float *src1, float *dst, void *stream) {
    TPRELU_f32_63x63<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 6: f32 1x16384
extern "C" __global__ AICORE void TPRELU_f32_1x16384(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTPRELU_f32_1x16384(float *src0, float *src1, float *dst, void *stream) {
    TPRELU_f32_1x16384<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 7: f32 2048x8
extern "C" __global__ AICORE void TPRELU_f32_2048x8(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTPRELU_f32_2048x8(float *src0, float *src1, float *dst, void *stream) {
    TPRELU_f32_2048x8<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}