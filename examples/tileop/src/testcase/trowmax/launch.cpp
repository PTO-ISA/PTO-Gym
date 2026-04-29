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

extern "C" __global__ AICORE void TROWMAX_f32_127x64_valid127x63(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_127x64_valid127x63(float *src, float *dst, void *stream) {
    TROWMAX_f32_127x64_valid127x63<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_63x64(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_63x64(float *src, float *dst, void *stream) {
    TROWMAX_f32_63x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_31x128_valid31x127(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_31x128_valid31x127(float *src, float *dst, void *stream) {
    TROWMAX_f32_31x128_valid31x127<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_15x192(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_15x192(float *src, float *dst, void *stream) {
    TROWMAX_f32_15x192<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_7x448_valid7x447(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_7x448_valid7x447(float *src, float *dst, void *stream) {
    TROWMAX_f32_7x448_valid7x447<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f16_256x16_valid256x15(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTROWMAX_f16_256x16_valid256x15(uint16_t *src, uint16_t *dst, void *stream) {
    TROWMAX_f16_256x16_valid256x15<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_30x216(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_30x216(float *src, float *dst, void *stream) {
    TROWMAX_f32_30x216<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_30x216_valid30x24(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_30x216_valid30x24(float *src, float *dst, void *stream) {
    TROWMAX_f32_30x216_valid30x24<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_30x216_valid11x216(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_30x216_valid11x216(float *src, float *dst, void *stream) {
    TROWMAX_f32_30x216_valid11x216<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_30x216_valid11x24(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_30x216_valid11x24(float *src, float *dst, void *stream) {
    TROWMAX_f32_30x216_valid11x24<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_238x40(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_238x40(float *src, float *dst, void *stream) {
    TROWMAX_f32_238x40<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_238x40_valid238x16(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_238x40_valid238x16(float *src, float *dst, void *stream) {
    TROWMAX_f32_238x40_valid238x16<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_238x40_valid121x40(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_238x40_valid121x40(float *src, float *dst, void *stream) {
    TROWMAX_f32_238x40_valid121x40<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_238x40_valid121x16(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_238x40_valid121x16(float *src, float *dst, void *stream) {
    TROWMAX_f32_238x40_valid121x16<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_64x128(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_64x128(float *src, float *dst, void *stream) {
    TROWMAX_f32_64x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_32x256(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_32x256(float *src, float *dst, void *stream) {
    TROWMAX_f32_32x256<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_16x512(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_16x512(float *src, float *dst, void *stream) {
    TROWMAX_f32_16x512<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWMAX_f32_8x1024(__gm__ float *src, __gm__ float *dst);

void LaunchTROWMAX_f32_8x1024(float *src, float *dst, void *stream) {
    TROWMAX_f32_8x1024<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// int32 cases
extern "C" __global__ AICORE void TROWMAX_i32_127x64_valid127x63(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTROWMAX_i32_127x64_valid127x63(int32_t *src, int32_t *dst, void *stream) {
    TROWMAX_i32_127x64_valid127x63<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i32_63x64(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTROWMAX_i32_63x64(int32_t *src, int32_t *dst, void *stream) {
    TROWMAX_i32_63x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i32_31x128_valid31x127(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTROWMAX_i32_31x128_valid31x127(int32_t *src, int32_t *dst, void *stream) {
    TROWMAX_i32_31x128_valid31x127<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i32_15x192(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTROWMAX_i32_15x192(int32_t *src, int32_t *dst, void *stream) {
    TROWMAX_i32_15x192<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i32_7x448_valid7x447(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTROWMAX_i32_7x448_valid7x447(int32_t *src, int32_t *dst, void *stream) {
    TROWMAX_i32_7x448_valid7x447<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

// int16 cases
extern "C" __global__ AICORE void TROWMAX_i16_128x64(__gm__ int16_t *src, __gm__ int16_t *dst);

void LaunchTROWMAX_i16_128x64(int16_t *src, int16_t *dst, void *stream) {
    TROWMAX_i16_128x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i16_64x64(__gm__ int16_t *src, __gm__ int16_t *dst);

void LaunchTROWMAX_i16_64x64(int16_t *src, int16_t *dst, void *stream) {
    TROWMAX_i16_64x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i16_32x128(__gm__ int16_t *src, __gm__ int16_t *dst);

void LaunchTROWMAX_i16_32x128(int16_t *src, int16_t *dst, void *stream) {
    TROWMAX_i16_32x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i16_16x192(__gm__ int16_t *src, __gm__ int16_t *dst);

void LaunchTROWMAX_i16_16x192(int16_t *src, int16_t *dst, void *stream) {
    TROWMAX_i16_16x192<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}

extern "C" __global__ AICORE void TROWMAX_i16_8x448(__gm__ int16_t *src, __gm__ int16_t *dst);

void LaunchTROWMAX_i16_8x448(int16_t *src, int16_t *dst, void *stream) {
    TROWMAX_i16_8x448<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}
