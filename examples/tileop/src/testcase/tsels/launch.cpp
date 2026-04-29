// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>
#include <cstring>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void TSELS_uint8_uint8_2x32_2x32_2x32_2x32(__gm__ uint8_t *mask, __gm__ uint8_t *src, __gm__ uint8_t *dst, uint8_t scalar);
void LaunchTSELS_uint8_uint8_2x32_2x32_2x32_2x32(uint8_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream) {
    uint8_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint8_t));
    TSELS_uint8_uint8_2x32_2x32_2x32_2x32<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint8_t *)src, (__gm__ uint8_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint8_uint16_2x32_2x16_2x32_2x32(__gm__ uint16_t *mask, __gm__ uint8_t *src, __gm__ uint8_t *dst, uint8_t scalar);
void LaunchTSELS_uint8_uint16_2x32_2x16_2x32_2x32(uint16_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream) {
    uint8_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint8_t));
    TSELS_uint8_uint16_2x32_2x16_2x32_2x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)mask, (__gm__ uint8_t *)src, (__gm__ uint8_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint8_uint32_2x32_2x8_2x32_2x32(__gm__ uint32_t *mask, __gm__ uint8_t *src, __gm__ uint8_t *dst, uint8_t scalar);
void LaunchTSELS_uint8_uint32_2x32_2x8_2x32_2x32(uint32_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream) {
    uint8_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint8_t));
    TSELS_uint8_uint32_2x32_2x8_2x32_2x32<<<1, nullptr, stream>>>((__gm__ uint32_t *)mask, (__gm__ uint8_t *)src, (__gm__ uint8_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint16_uint8_2x16_2x32_2x16_2x16(__gm__ uint8_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_uint16_uint8_2x16_2x32_2x16_2x16(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_uint16_uint8_2x16_2x32_2x16_2x16<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint16_uint16_2x16_2x16_2x16_2x16(__gm__ uint16_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_uint16_uint16_2x16_2x16_2x16_2x16(uint16_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_uint16_uint16_2x16_2x16_2x16_2x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint16_uint32_2x16_2x8_2x16_2x16(__gm__ uint32_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_uint16_uint32_2x16_2x8_2x16_2x16(uint32_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_uint16_uint32_2x16_2x8_2x16_2x16<<<1, nullptr, stream>>>((__gm__ uint32_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint32_uint8_2x8_2x32_2x8_2x8(__gm__ uint8_t *mask, __gm__ uint32_t *src, __gm__ uint32_t *dst, uint32_t scalar);
void LaunchTSELS_uint32_uint8_2x8_2x32_2x8_2x8(uint8_t *mask, uint32_t *src, uint32_t *dst, void *scalar_ptr, void *stream) {
    uint32_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint32_t));
    TSELS_uint32_uint8_2x8_2x32_2x8_2x8<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint32_t *)src, (__gm__ uint32_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint32_uint16_2x8_2x16_2x8_2x8(__gm__ uint16_t *mask, __gm__ uint32_t *src, __gm__ uint32_t *dst, uint32_t scalar);
void LaunchTSELS_uint32_uint16_2x8_2x16_2x8_2x8(uint16_t *mask, uint32_t *src, uint32_t *dst, void *scalar_ptr, void *stream) {
    uint32_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint32_t));
    TSELS_uint32_uint16_2x8_2x16_2x8_2x8<<<1, nullptr, stream>>>((__gm__ uint16_t *)mask, (__gm__ uint32_t *)src, (__gm__ uint32_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint32_uint32_2x8_2x8_2x8_2x8(__gm__ uint32_t *mask, __gm__ uint32_t *src, __gm__ uint32_t *dst, uint32_t scalar);
void LaunchTSELS_uint32_uint32_2x8_2x8_2x8_2x8(uint32_t *mask, uint32_t *src, uint32_t *dst, void *scalar_ptr, void *stream) {
    uint32_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint32_t));
    TSELS_uint32_uint32_2x8_2x8_2x8_2x8<<<1, nullptr, stream>>>((__gm__ uint32_t *)mask, (__gm__ uint32_t *)src, (__gm__ uint32_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f16_uint8_2x16_2x32_2x16_2x16(__gm__ uint8_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_f16_uint8_2x16_2x32_2x16_2x16(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_f16_uint8_2x16_2x32_2x16_2x16<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f16_uint16_2x16_2x16_2x16_2x16(__gm__ uint16_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_f16_uint16_2x16_2x16_2x16_2x16(uint16_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_f16_uint16_2x16_2x16_2x16_2x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f16_uint32_2x16_2x8_2x16_2x16(__gm__ uint32_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_f16_uint32_2x16_2x8_2x16_2x16(uint32_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_f16_uint32_2x16_2x8_2x16_2x16<<<1, nullptr, stream>>>((__gm__ uint32_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f32_uint8_2x8_2x32_2x8_2x8(__gm__ uint8_t *mask, __gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTSELS_f32_uint8_2x8_2x32_2x8_2x8(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream) {
    float scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(float));
    TSELS_f32_uint8_2x8_2x32_2x8_2x8<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src, (__gm__ float *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f32_uint16_2x8_2x16_2x8_2x8(__gm__ uint16_t *mask, __gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTSELS_f32_uint16_2x8_2x16_2x8_2x8(uint16_t *mask, float *src, float *dst, void *scalar_ptr, void *stream) {
    float scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(float));
    TSELS_f32_uint16_2x8_2x16_2x8_2x8<<<1, nullptr, stream>>>((__gm__ uint16_t *)mask, (__gm__ float *)src, (__gm__ float *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f32_uint32_2x8_2x8_2x8_2x8(__gm__ uint32_t *mask, __gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTSELS_f32_uint32_2x8_2x8_2x8_2x8(uint32_t *mask, float *src, float *dst, void *scalar_ptr, void *stream) {
    float scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(float));
    TSELS_f32_uint32_2x8_2x8_2x8_2x8<<<1, nullptr, stream>>>((__gm__ uint32_t *)mask, (__gm__ float *)src, (__gm__ float *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint8_uint8_2x32_2x64_2x128_2x31(__gm__ uint8_t *mask, __gm__ uint8_t *src, __gm__ uint8_t *dst, uint8_t scalar);
void LaunchTSELS_uint8_uint8_2x32_2x64_2x128_2x31(uint8_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream) {
    uint8_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint8_t));
    TSELS_uint8_uint8_2x32_2x64_2x128_2x31<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint8_t *)src, (__gm__ uint8_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint16_uint8_2x32_2x64_2x128_2x31(__gm__ uint8_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_uint16_uint8_2x32_2x64_2x128_2x31(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_uint16_uint8_2x32_2x64_2x128_2x31<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f32_uint8_2x32_2x64_2x128_2x31(__gm__ uint8_t *mask, __gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTSELS_f32_uint8_2x32_2x64_2x128_2x31(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream) {
    float scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(float));
    TSELS_f32_uint8_2x32_2x64_2x128_2x31<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src, (__gm__ float *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_uint8_uint8_32x672_32x96_32x672_32x666(__gm__ uint8_t *mask, __gm__ uint8_t *src, __gm__ uint8_t *dst, uint8_t scalar);
void LaunchTSELS_uint8_uint8_32x672_32x96_32x672_32x666(uint8_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream) {
    uint8_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint8_t));
    TSELS_uint8_uint8_32x672_32x96_32x672_32x666<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint8_t *)src, (__gm__ uint8_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f16_uint8_32x672_32x96_32x672_32x666(__gm__ uint8_t *mask, __gm__ uint16_t *src, __gm__ uint16_t *dst, uint16_t scalar);
void LaunchTSELS_f16_uint8_32x672_32x96_32x672_32x666(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream) {
    uint16_t scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(uint16_t));
    TSELS_f16_uint8_32x672_32x96_32x672_32x666<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f32_uint8_32x672_32x96_32x672_32x666(__gm__ uint8_t *mask, __gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTSELS_f32_uint8_32x672_32x96_32x672_32x666(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream) {
    float scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(float));
    TSELS_f32_uint8_32x672_32x96_32x672_32x666<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src, (__gm__ float *)dst, scalar);
}

extern "C" __global__ AICORE void TSELS_f32_uint8_1x8192_1x4096_1x8192_1x8192(__gm__ uint8_t *mask, __gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTSELS_f32_uint8_1x8192_1x4096_1x8192_1x8192(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream) {
    float scalar;
    std::memcpy(&scalar, scalar_ptr, sizeof(float));
    TSELS_f32_uint8_1x8192_1x4096_1x8192_1x8192<<<1, nullptr, stream>>>((__gm__ uint8_t *)mask, (__gm__ float *)src, (__gm__ float *)dst, scalar);
}