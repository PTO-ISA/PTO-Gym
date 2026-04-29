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

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_8x1_8x8_8x8(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_8x1_8x8_8x8(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_8x1_8x8_8x8<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_1024x1_1024x8_1024x8(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_1024x1_1024x8_1024x8(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_1024x1_1024x8_1024x8<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_16x1_13x16_13x13(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_16x1_13x16_13x13(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_16x1_13x16_13x13<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_1024x1_1023x24_1023x17(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_1024x1_1023x24_1023x17(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_1024x1_1023x24_1023x17<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_8x1_8x64_8x64(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_8x1_8x64_8x64(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_8x1_8x64_8x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_264x1_260x64_260x64(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_264x1_260x64_260x64(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_264x1_260x64_260x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_8x1_1x128_1x128(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_8x1_1x128_1x128(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_8x1_1x128_1x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_64x1_32x128_32x128(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_64x1_32x128_32x128(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_64x1_32x128_32x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_8x1_3x4096_3x4095(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_8x1_3x4096_3x4095(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_8x1_3x4096_3x4095<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_8x1_2x16384_2x16381(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_8x1_2x16384_2x16381(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_8x1_2x16384_2x16381<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_16x1_2x16_2x16(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_16x1_2x16_2x16(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_16x1_2x16_2x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_16x1_13x16_13x13(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_16x1_13x16_13x13(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_16x1_13x16_13x13<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_272x1_260x64_260x64(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_272x1_260x64_260x64(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_272x1_260x64_260x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_16x1_3x8192_3x8191(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_16x1_3x8192_3x8191(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_16x1_3x8192_3x8191<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_16x1_1x16384_1x16381(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_16x1_1x16384_1x16381(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_16x1_1x16384_1x16381<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_16x1_1x32768_1x32761(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_16x1_1x32768_1x32761(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_16x1_1x32768_1x32761<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_int32_float_16x1_13x16_13x13(__gm__ float *src, __gm__ int32_t *dst);
void LaunchTROWARGMAX_int32_float_16x1_13x16_13x13(float *src, int32_t *dst, void *stream) {
    TROWARGMAX_int32_float_16x1_13x16_13x13<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_int32_half_16x1_13x16_13x13(__gm__ uint16_t *src, __gm__ int32_t *dst);
void LaunchTROWARGMAX_int32_half_16x1_13x16_13x13(uint16_t *src, int32_t *dst, void *stream) {
    TROWARGMAX_int32_half_16x1_13x16_13x13<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_3x8_3x3480_3x3473(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_3x8_3x3480_3x3473(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_3x8_3x3480_3x3473<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_260x8_260x64_260x64(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_260x8_260x64_260x64(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_260x8_260x64_260x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_float_1023x8_1023x24_1023x17(__gm__ float *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_float_1023x8_1023x24_1023x17(float *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_float_1023x8_1023x24_1023x17<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_3x16_3x3488_3x3473(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_3x16_3x3488_3x3473(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_3x16_3x3488_3x3473<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_260x16_260x64_260x64(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_260x16_260x64_260x64(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_260x16_260x64_260x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}

extern "C" __global__ AICORE void TROWARGMAX_uint32_half_1023x16_1023x32_1023x17(__gm__ uint16_t *src, __gm__ uint32_t *dst);
void LaunchTROWARGMAX_uint32_half_1023x16_1023x32_1023x17(uint16_t *src, uint32_t *dst, void *stream) {
    TROWARGMAX_uint32_half_1023x16_1023x32_1023x17<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint32_t *)dst);
}
