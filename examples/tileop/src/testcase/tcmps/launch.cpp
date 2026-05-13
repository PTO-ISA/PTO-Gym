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

// Scalar value for comparison (must match gen_data.py SCALAR)
static constexpr float TCMP_SCALAR_F32 = 5.0f;
static constexpr int32_t TCMP_SCALAR_I32 = 5;

// Case 0: f32 1x64
extern "C" __global__ AICORE void TCMP_f32_1x64(__gm__ float *src, __gm__ uint8_t *dst, float scalar);

void LaunchTCMP_f32_1x64(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_1x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}

// Case 1: f32 4x64
extern "C" __global__ AICORE void TCMP_f32_4x64(__gm__ float *src, __gm__ uint8_t *dst, float scalar);

void LaunchTCMP_f32_4x64(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_4x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}

// Case 2: f32 8x64
extern "C" __global__ AICORE void TCMP_f32_8x64(__gm__ float *src, __gm__ uint8_t *dst, float scalar);

void LaunchTCMP_f32_8x64(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_8x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}

// Case 3: f32 32x64
extern "C" __global__ AICORE void TCMP_f32_32x64(__gm__ float *src, __gm__ uint8_t *dst, float scalar);

void LaunchTCMP_f32_32x64(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_32x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}

// Case 4: f32 128x128
extern "C" __global__ AICORE void TCMP_f32_128x128(__gm__ float *src, __gm__ uint8_t *dst, float scalar);

void LaunchTCMP_f32_128x128(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_128x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}

// Case 5: i32 16x32
extern "C" __global__ AICORE void TCMP_i32_16x32(__gm__ int32_t *src, __gm__ uint8_t *dst, int32_t scalar);

void LaunchTCMP_i32_16x32(int32_t *src, uint8_t *dst, void *stream) {
    TCMP_i32_16x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_I32);
}

// Case 6: i32 32x32
extern "C" __global__ AICORE void TCMP_i32_32x32(__gm__ int32_t *src, __gm__ uint8_t *dst, int32_t scalar);

void LaunchTCMP_i32_32x32(int32_t *src, uint8_t *dst, void *stream) {
    TCMP_i32_32x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_I32);
}

// Case 7: i32 64x64 tile with valid 32x64
extern "C" __global__ AICORE void TCMP_i32_32x64_valid32x64(__gm__ int32_t *src, __gm__ uint8_t *dst, int32_t scalar);

void LaunchTCMP_i32_32x64_valid32x64(int32_t *src, uint8_t *dst, void *stream) {
    TCMP_i32_32x64_valid32x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_I32);
}

// Case 8: f32 7x448
extern "C" __global__ AICORE void TCMP_f32_7x448(__gm__ float *src, __gm__ uint8_t *dst, float scalar);

void LaunchTCMP_f32_7x448(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_7x448<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}

// Case 9: f32 256x16
extern "C" __global__ AICORE void TCMP_f32_256x16(__gm__ float *src, __gm__ uint8_t *dst, float scalar);

void LaunchTCMP_f32_256x16(float *src, uint8_t *dst, void *stream) {
    TCMP_f32_256x16<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F32);
}

// Case 10: i32 31x128
extern "C" __global__ AICORE void TCMP_i32_31x128(__gm__ int32_t *src, __gm__ uint8_t *dst, int32_t scalar);

void LaunchTCMP_i32_31x128(int32_t *src, uint8_t *dst, void *stream) {
    TCMP_i32_31x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_I32);
}

// Case 11: f16 32x128
static constexpr uint16_t TCMP_SCALAR_F16 = 0x4500; // 5.0 in half precision

extern "C" __global__ AICORE void TCMP_f16_32x128(__gm__ uint16_t *src, __gm__ uint8_t *dst, uint16_t scalar);

void LaunchTCMP_f16_32x128(uint16_t *src, uint8_t *dst, void *stream) {
    TCMP_f16_32x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_F16);
}

// Case 12: i16 32x128
static constexpr int16_t TCMP_SCALAR_I16 = 5;

extern "C" __global__ AICORE void TCMP_i16_32x128(__gm__ int16_t *src, __gm__ uint8_t *dst, int16_t scalar);

void LaunchTCMP_i16_32x128(int16_t *src, uint8_t *dst, void *stream) {
    TCMP_i16_32x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst, TCMP_SCALAR_I16);
}
