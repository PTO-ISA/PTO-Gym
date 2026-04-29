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

// Case 0: f32 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLSUM_f32_1x256(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLSUM_f32_1x256(float *dst, float *src, void *stream) {
    TCOLSUM_f32_1x256<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 1: f32 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_f32_16x128(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLSUM_f32_16x128(float *dst, float *src, void *stream) {
    TCOLSUM_f32_16x128<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 2: f32 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLSUM_f32_16x256(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLSUM_f32_16x256(float *dst, float *src, void *stream) {
    TCOLSUM_f32_16x256<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 3: f32 64x128_1 (input: 64x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_f32_64x128_1(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLSUM_f32_64x128_1(float *dst, float *src, void *stream) {
    TCOLSUM_f32_64x128_1<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 4: f32 64x128_2 (input: 64x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_f32_64x128_2(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLSUM_f32_64x128_2(float *dst, float *src, void *stream) {
    TCOLSUM_f32_64x128_2<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 5: f32 1x512 (input: 1x512, output: 1x512)
extern "C" __global__ AICORE void TCOLSUM_f32_1x512(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLSUM_f32_1x512(float *dst, float *src, void *stream) {
    TCOLSUM_f32_1x512<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 6: f16 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLSUM_f16_1x256(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLSUM_f16_1x256(void *dst, void *src, void *stream) {
    TCOLSUM_f16_1x256<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 7: f16 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_f16_16x128(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLSUM_f16_16x128(void *dst, void *src, void *stream) {
    TCOLSUM_f16_16x128<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 8: f16 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLSUM_f16_16x256(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLSUM_f16_16x256(void *dst, void *src, void *stream) {
    TCOLSUM_f16_16x256<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 9: f16 64x128_1 (input: 64x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_f16_64x128_1(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLSUM_f16_64x128_1(void *dst, void *src, void *stream) {
    TCOLSUM_f16_64x128_1<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 10: f16 64x128_2 (input: 64x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_f16_64x128_2(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLSUM_f16_64x128_2(void *dst, void *src, void *stream) {
    TCOLSUM_f16_64x128_2<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 11: i8 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLSUM_i8_1x256(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLSUM_i8_1x256(void *dst, void *src, void *stream) {
    TCOLSUM_i8_1x256<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}

// Case 12: i8 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_i8_16x128(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLSUM_i8_16x128(void *dst, void *src, void *stream) {
    TCOLSUM_i8_16x128<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}

// Case 13: i8 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLSUM_i8_16x256(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLSUM_i8_16x256(void *dst, void *src, void *stream) {
    TCOLSUM_i8_16x256<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}

// Case 14: i8 64x128_1 (input: 64x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_i8_64x128_1(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLSUM_i8_64x128_1(void *dst, void *src, void *stream) {
    TCOLSUM_i8_64x128_1<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}

// Case 15: i8 64x128_2 (input: 64x128, output: 1x128)
extern "C" __global__ AICORE void TCOLSUM_i8_64x128_2(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLSUM_i8_64x128_2(void *dst, void *src, void *stream) {
    TCOLSUM_i8_64x128_2<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}