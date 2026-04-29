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

extern "C" __global__ AICORE void TROWPROD_f32_127x64_valid127x63(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_127x64_valid127x63(float *src, float *dst, void *stream) {
    TROWPROD_f32_127x64_valid127x63<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_63x64(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_63x64(float *src, float *dst, void *stream) {
    TROWPROD_f32_63x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_31x128_valid31x127(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_31x128_valid31x127(float *src, float *dst, void *stream) {
    TROWPROD_f32_31x128_valid31x127<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_15x192(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_15x192(float *src, float *dst, void *stream) {
    TROWPROD_f32_15x192<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_7x448_valid7x447(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_7x448_valid7x447(float *src, float *dst, void *stream) {
    TROWPROD_f32_7x448_valid7x447<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f16_256x16_valid256x15(__gm__ uint16_t *src, __gm__ uint16_t *dst);
void LaunchTROWPROD_f16_256x16_valid256x15(uint16_t *src, uint16_t *dst, void *stream) {
    TROWPROD_f16_256x16_valid256x15<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_64x128(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_64x128(float *src, float *dst, void *stream) {
    TROWPROD_f32_64x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_32x256(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_32x256(float *src, float *dst, void *stream) {
    TROWPROD_f32_32x256<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_16x512(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_16x512(float *src, float *dst, void *stream) {
    TROWPROD_f32_16x512<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TROWPROD_f32_8x1024(__gm__ float *src, __gm__ float *dst);
void LaunchTROWPROD_f32_8x1024(float *src, float *dst, void *stream) {
    TROWPROD_f32_8x1024<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// int32 cases
extern "C" __global__ AICORE void TROWPROD_i32_127x64_valid127x63(__gm__ int32_t *src, __gm__ int32_t *dst);
void LaunchTROWPROD_i32_127x64_valid127x63(int32_t *src, int32_t *dst, void *stream) {
    TROWPROD_i32_127x64_valid127x63<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWPROD_i32_63x64(__gm__ int32_t *src, __gm__ int32_t *dst);
void LaunchTROWPROD_i32_63x64(int32_t *src, int32_t *dst, void *stream) {
    TROWPROD_i32_63x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWPROD_i32_31x128_valid31x127(__gm__ int32_t *src, __gm__ int32_t *dst);
void LaunchTROWPROD_i32_31x128_valid31x127(int32_t *src, int32_t *dst, void *stream) {
    TROWPROD_i32_31x128_valid31x127<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWPROD_i32_15x192(__gm__ int32_t *src, __gm__ int32_t *dst);
void LaunchTROWPROD_i32_15x192(int32_t *src, int32_t *dst, void *stream) {
    TROWPROD_i32_15x192<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWPROD_i32_7x448_valid7x447(__gm__ int32_t *src, __gm__ int32_t *dst);
void LaunchTROWPROD_i32_7x448_valid7x447(int32_t *src, int32_t *dst, void *stream) {
    TROWPROD_i32_7x448_valid7x447<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

// int16 cases
extern "C" __global__ AICORE void TROWPROD_i16_256x16_valid256x15(__gm__ int16_t *src, __gm__ int16_t *dst);
void LaunchTROWPROD_i16_256x16_valid256x15(int16_t *src, int16_t *dst, void *stream) {
    TROWPROD_i16_256x16_valid256x15<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}

extern "C" __global__ AICORE void TROWPROD_i16_63x64(__gm__ int16_t *src, __gm__ int16_t *dst);
void LaunchTROWPROD_i16_63x64(int16_t *src, int16_t *dst, void *stream) {
    TROWPROD_i16_63x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}

extern "C" __global__ AICORE void TROWPROD_i16_31x128_valid31x127(__gm__ int16_t *src, __gm__ int16_t *dst);
void LaunchTROWPROD_i16_31x128_valid31x127(int16_t *src, int16_t *dst, void *stream) {
    TROWPROD_i16_31x128_valid31x127<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}
