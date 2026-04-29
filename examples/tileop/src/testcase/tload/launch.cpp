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

extern "C" __global__ AICORE void TLOAD_ND_f32_16x64(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TLOAD_DN_f32_16x64(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TLOAD_NZ_f32_128x128(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TLOAD_ND_PAD_ZERO_f32_16x64(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TLOAD_DN_PAD_MAX_f32_16x64(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TLOAD_NZ_PAD_MIN_f32_128x128(__gm__ float *src, __gm__ float *dst);

void LaunchTLOAD_ND_f32_16x64(float *src, float *dst, void *stream) {
    TLOAD_ND_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTLOAD_DN_f32_16x64(float *src, float *dst, void *stream) {
    TLOAD_DN_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTLOAD_NZ_f32_128x128(float *src, float *dst, void *stream) {
    TLOAD_NZ_f32_128x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTLOAD_ND_PAD_ZERO_f32_16x64(float *src, float *dst, void *stream) {
    TLOAD_ND_PAD_ZERO_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTLOAD_DN_PAD_MAX_f32_16x64(float *src, float *dst, void *stream) {
    TLOAD_DN_PAD_MAX_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTLOAD_NZ_PAD_MIN_f32_128x128(float *src, float *dst, void *stream) {
    TLOAD_NZ_PAD_MIN_f32_128x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
