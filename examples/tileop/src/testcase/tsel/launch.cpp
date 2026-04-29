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

// Case 0: f32 2x128
extern "C" __global__ AICORE void TSEL_f32_2x128(__gm__ uint8_t *mask, __gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTSEL_f32_2x128(uint8_t *mask, float *src0, float *src1, float *dst, void *stream) {
    TSEL_f32_2x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 1: f32 2x32
extern "C" __global__ AICORE void TSEL_f32_2x32(__gm__ uint8_t *mask, __gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTSEL_f32_2x32(uint8_t *mask, float *src0, float *src1, float *dst, void *stream) {
    TSEL_f32_2x32<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 2: f32 2x160
extern "C" __global__ AICORE void TSEL_f32_2x160(__gm__ uint8_t *mask, __gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTSEL_f32_2x160(uint8_t *mask, float *src0, float *src1, float *dst, void *stream) {
    TSEL_f32_2x160<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 3: f32 2x512
extern "C" __global__ AICORE void TSEL_f32_2x512(__gm__ uint8_t *mask, __gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTSEL_f32_2x512(uint8_t *mask, float *src0, float *src1, float *dst, void *stream) {
    TSEL_f32_2x512<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 4: f16 2x128
extern "C" __global__ AICORE void TSEL_f16_2x128(__gm__ uint8_t *mask, __gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTSEL_f16_2x128(uint8_t *mask, uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TSEL_f16_2x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 5: f16 2x32
extern "C" __global__ AICORE void TSEL_f16_2x32(__gm__ uint8_t *mask, __gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTSEL_f16_2x32(uint8_t *mask, uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TSEL_f16_2x32<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 6: f16 2x160
extern "C" __global__ AICORE void TSEL_f16_2x160(__gm__ uint8_t *mask, __gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTSEL_f16_2x160(uint8_t *mask, uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TSEL_f16_2x160<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 7: i8 2x128
extern "C" __global__ AICORE void TSEL_i8_2x128(__gm__ uint8_t *mask, __gm__ int8_t *src0, __gm__ int8_t *src1, __gm__ int8_t *dst);

void LaunchTSEL_i8_2x128(uint8_t *mask, int8_t *src0, int8_t *src1, int8_t *dst, void *stream) {
    TSEL_i8_2x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ int8_t *)src0, (__gm__ int8_t *)src1, (__gm__ int8_t *)dst);
}

// Case 8: i8 2x32
extern "C" __global__ AICORE void TSEL_i8_2x32(__gm__ uint8_t *mask, __gm__ int8_t *src0, __gm__ int8_t *src1, __gm__ int8_t *dst);

void LaunchTSEL_i8_2x32(uint8_t *mask, int8_t *src0, int8_t *src1, int8_t *dst, void *stream) {
    TSEL_i8_2x32<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ int8_t *)src0, (__gm__ int8_t *)src1, (__gm__ int8_t *)dst);
}

// Case 9: i8 2x160
extern "C" __global__ AICORE void TSEL_i8_2x160(__gm__ uint8_t *mask, __gm__ int8_t *src0, __gm__ int8_t *src1, __gm__ int8_t *dst);

void LaunchTSEL_i8_2x160(uint8_t *mask, int8_t *src0, int8_t *src1, int8_t *dst, void *stream) {
    TSEL_i8_2x160<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ int8_t *)src0, (__gm__ int8_t *)src1, (__gm__ int8_t *)dst);
}