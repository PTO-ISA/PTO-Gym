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

// Case: f32 1x32
extern "C" __global__ AICORE void TSORT32_f32_1x32(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_1x32(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_1x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 1x64
extern "C" __global__ AICORE void TSORT32_f32_1x64(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_1x64(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_1x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 2x32
extern "C" __global__ AICORE void TSORT32_f32_2x32(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_2x32(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_2x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 16x32
extern "C" __global__ AICORE void TSORT32_f32_16x32(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_16x32(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_16x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 2x64 shared_idx
extern "C" __global__ AICORE void TSORT32_f32_2x64_shared_idx(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_2x64_shared_idx(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_2x64_shared_idx<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 16x64 shared_idx
extern "C" __global__ AICORE void TSORT32_f32_16x64_shared_idx(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_16x64_shared_idx(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_16x64_shared_idx<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 1x8192 (large shape)
extern "C" __global__ AICORE void TSORT32_f32_1x8192(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_1x8192(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_1x8192<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f16 1x32
extern "C" __global__ AICORE void TSORT32_f16_1x32(__gm__ uint16_t *src, __gm__ uint32_t *idx, __gm__ uint16_t *dst);

void LaunchTSORT32_f16_1x32(uint16_t *src, uint32_t *idx, uint16_t *dst, void *stream) {
    TSORT32_f16_1x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)idx, (__gm__ uint16_t *)dst);
}

// Case: f16 4x64
extern "C" __global__ AICORE void TSORT32_f16_4x64(__gm__ uint16_t *src, __gm__ uint32_t *idx, __gm__ uint16_t *dst);

void LaunchTSORT32_f16_4x64(uint16_t *src, uint32_t *idx, uint16_t *dst, void *stream) {
    TSORT32_f16_4x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)idx, (__gm__ uint16_t *)dst);
}

// Case: f32 2x13 (non-32-aligned, requires tmp buffer for padding)
extern "C" __global__ AICORE void TSORT32_f32_2x13(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_2x13(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_2x13<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 1x4164 (non-32-aligned large shape)
extern "C" __global__ AICORE void TSORT32_f32_1x4164(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_1x4164(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_1x4164<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}

// Case: f32 2x2084 (non-32-aligned multi-row shape)
extern "C" __global__ AICORE void TSORT32_f32_2x2084(__gm__ float *src, __gm__ uint32_t *idx, __gm__ float *dst);

void LaunchTSORT32_f32_2x2084(float *src, uint32_t *idx, float *dst, void *stream) {
    TSORT32_f32_2x2084<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)idx, (__gm__ float *)dst);
}