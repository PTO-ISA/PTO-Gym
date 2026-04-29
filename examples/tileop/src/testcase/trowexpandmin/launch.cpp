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

// f32 kernels
extern "C" __global__ AICORE void TROWEXPANDMIN_f32_16x32(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);
extern "C" __global__ AICORE void TROWEXPANDMIN_f32_56x128(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTROWEXPANDMIN_f32_16x32(float *src0, float *src1, float *dst, void *stream) {
    TROWEXPANDMIN_f32_16x32<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}
void LaunchTROWEXPANDMIN_f32_56x128(float *src0, float *src1, float *dst, void *stream) {
    TROWEXPANDMIN_f32_56x128<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// f16 kernels (use uint16_t for aclFloat16)
extern "C" __global__ AICORE void TROWEXPANDMIN_f16_48x64(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TROWEXPANDMIN_f16_16x128(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TROWEXPANDMIN_f16_32x64_noeq(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTROWEXPANDMIN_f16_48x64(void *src0, void *src1, void *dst, void *stream) {
    TROWEXPANDMIN_f16_48x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}
void LaunchTROWEXPANDMIN_f16_16x128(void *src0, void *src1, void *dst, void *stream) {
    TROWEXPANDMIN_f16_16x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}
void LaunchTROWEXPANDMIN_f16_32x64_noeq(void *src0, void *src1, void *dst, void *stream) {
    TROWEXPANDMIN_f16_32x64_noeq<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// i32 kernels
extern "C" __global__ AICORE void TROWEXPANDMIN_i32_16x32(__gm__ int32_t *src0, __gm__ int32_t *src1, __gm__ int32_t *dst);

void LaunchTROWEXPANDMIN_i32_16x32(void *src0, void *src1, void *dst, void *stream) {
    TROWEXPANDMIN_i32_16x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src0, (__gm__ int32_t *)src1, (__gm__ int32_t *)dst);
}

// i16 kernels
extern "C" __global__ AICORE void TROWEXPANDMIN_i16_16x64(__gm__ int16_t *src0, __gm__ int16_t *src1, __gm__ int16_t *dst);

void LaunchTROWEXPANDMIN_i16_16x64(void *src0, void *src1, void *dst, void *stream) {
    TROWEXPANDMIN_i16_16x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src0, (__gm__ int16_t *)src1, (__gm__ int16_t *)dst);
}

// Note: launchTRowExpandMin2 with src1Col>1 has different semantics - TBD