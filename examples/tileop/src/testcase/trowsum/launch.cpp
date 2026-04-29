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

// ========================================================================
// f32 kernels (case1-case10)
// ========================================================================

extern "C" __global__ AICORE void TROWSUM_case1(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case2(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case3(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case4(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case5(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case6(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TROWSUM_case7(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case8(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case9(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWSUM_case10(__gm__ float *src, __gm__ float *dst);

void LaunchTROWSUM_case1(float *src, float *dst, void *stream) {
    TROWSUM_case1<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case2(float *src, float *dst, void *stream) {
    TROWSUM_case2<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case3(float *src, float *dst, void *stream) {
    TROWSUM_case3<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case4(float *src, float *dst, void *stream) {
    TROWSUM_case4<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case5(float *src, float *dst, void *stream) {
    TROWSUM_case5<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case6(uint16_t *src, uint16_t *dst, void *stream) {
    TROWSUM_case6<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}
void LaunchTROWSUM_case7(float *src, float *dst, void *stream) {
    TROWSUM_case7<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case8(float *src, float *dst, void *stream) {
    TROWSUM_case8<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case9(float *src, float *dst, void *stream) {
    TROWSUM_case9<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWSUM_case10(float *src, float *dst, void *stream) {
    TROWSUM_case10<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// ========================================================================
// i32 kernels (case11-case15)
// ========================================================================

extern "C" __global__ AICORE void TROWSUM_case11(__gm__ int32_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TROWSUM_case12(__gm__ int32_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TROWSUM_case13(__gm__ int32_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TROWSUM_case14(__gm__ int32_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TROWSUM_case15(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTROWSUM_case11(int32_t *src, int32_t *dst, void *stream) {
    TROWSUM_case11<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}
void LaunchTROWSUM_case12(int32_t *src, int32_t *dst, void *stream) {
    TROWSUM_case12<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}
void LaunchTROWSUM_case13(int32_t *src, int32_t *dst, void *stream) {
    TROWSUM_case13<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}
void LaunchTROWSUM_case14(int32_t *src, int32_t *dst, void *stream) {
    TROWSUM_case14<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}
void LaunchTROWSUM_case15(int32_t *src, int32_t *dst, void *stream) {
    TROWSUM_case15<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

// ========================================================================
// i16 kernels (case16-case20)
// ========================================================================

extern "C" __global__ AICORE void TROWSUM_case16(__gm__ int16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TROWSUM_case17(__gm__ int16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TROWSUM_case18(__gm__ int16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TROWSUM_case19(__gm__ int16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TROWSUM_case20(__gm__ int16_t *src, __gm__ int16_t *dst);

void LaunchTROWSUM_case16(int16_t *src, int16_t *dst, void *stream) {
    TROWSUM_case16<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}
void LaunchTROWSUM_case17(int16_t *src, int16_t *dst, void *stream) {
    TROWSUM_case17<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}
void LaunchTROWSUM_case18(int16_t *src, int16_t *dst, void *stream) {
    TROWSUM_case18<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}
void LaunchTROWSUM_case19(int16_t *src, int16_t *dst, void *stream) {
    TROWSUM_case19<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}
void LaunchTROWSUM_case20(int16_t *src, int16_t *dst, void *stream) {
    TROWSUM_case20<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}
