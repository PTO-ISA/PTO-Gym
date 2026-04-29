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

// Case 0: f32 64x64
extern "C" __global__ AICORE void TMIN_f32_64x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTMIN_f32_64x64(void *a, void *b, void *c, void *stream) {
    TMIN_f32_64x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 1: i32 64x64
extern "C" __global__ AICORE void TMIN_i32_64x64(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTMIN_i32_64x64(void *a, void *b, void *c, void *stream) {
    TMIN_i32_64x64<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}

// Case 2: i16 64x64
extern "C" __global__ AICORE void TMIN_i16_64x64(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTMIN_i16_64x64(void *a, void *b, void *c, void *stream) {
    TMIN_i16_64x64<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}

// Case 3: f16 64x64
extern "C" __global__ AICORE void TMIN_f16_64x64(__gm__ half *a, __gm__ half *b, __gm__ half *c);

void LaunchTMIN_f16_64x64(void *a, void *b, void *c, void *stream) {
    TMIN_f16_64x64<<<1, nullptr, stream>>>((__gm__ half *)a, (__gm__ half *)b, (__gm__ half *)c);
}

// Case 4: f32 64x64 v60x60
extern "C" __global__ AICORE void TMIN_f32_64x64_v60x60(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTMIN_f32_64x64_v60x60(void *a, void *b, void *c, void *stream) {
    TMIN_f32_64x64_v60x60<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 5: i32 64x64 v60x60
extern "C" __global__ AICORE void TMIN_i32_64x64_v60x60(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTMIN_i32_64x64_v60x60(void *a, void *b, void *c, void *stream) {
    TMIN_i32_64x64_v60x60<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}

// Case 6: f16 2x4096 v1x3600
extern "C" __global__ AICORE void TMIN_f16_2x4096_v1x3600(__gm__ half *a, __gm__ half *b, __gm__ half *c);

void LaunchTMIN_f16_2x4096_v1x3600(void *a, void *b, void *c, void *stream) {
    TMIN_f16_2x4096_v1x3600<<<1, nullptr, stream>>>((__gm__ half *)a, (__gm__ half *)b, (__gm__ half *)c);
}

// Case 7: i16 20x512 v16x200
extern "C" __global__ AICORE void TMIN_i16_20x512_v16x200(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTMIN_i16_20x512_v16x200(void *a, void *b, void *c, void *stream) {
    TMIN_i16_20x512_v16x200<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}