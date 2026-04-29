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

static constexpr float TDIVS_SCALAR_F32 = 3.0f;

// ========== src / scalar direction ==========

// Case 0: f32 32x64
extern "C" __global__ AICORE void TDIVS_f32_32x64(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_32x64(float *src, float *dst, void *stream) {
    TDIVS_f32_32x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 1: f16 63x64
extern "C" __global__ AICORE void TDIVS_f16_63x64(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_63x64(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_63x64<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x4200);
}

// Case 2: f32 7x448
extern "C" __global__ AICORE void TDIVS_f32_7x448(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_7x448(float *src, float *dst, void *stream) {
    TDIVS_f32_7x448<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 3: f32 256x16
extern "C" __global__ AICORE void TDIVS_f32_256x16(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_256x16(float *src, float *dst, void *stream) {
    TDIVS_f32_256x16<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// ========== scalar / src direction ==========

// Case 4: f32 32x64 scalar/src
extern "C" __global__ AICORE void TDIVS_f32_32x64_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_32x64_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_32x64_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 5: f16 63x64 scalar/src
extern "C" __global__ AICORE void TDIVS_f16_63x64_scalar_src(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_63x64_scalar_src(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_63x64_scalar_src<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x4200);
}

// Case 6: f32 7x448 scalar/src
extern "C" __global__ AICORE void TDIVS_f32_7x448_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_7x448_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_7x448_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 7: f32 256x16 scalar/src
extern "C" __global__ AICORE void TDIVS_f32_256x16_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_256x16_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_256x16_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}
