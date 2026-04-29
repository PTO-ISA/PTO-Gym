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

// ========== float32 kernels ==========

extern "C" __global__ AICORE void TEXPANDS_f32_16x64_scalar5(__gm__ float *dst);
extern "C" __global__ AICORE void TEXPANDS_f32_32x32_scalar3(__gm__ float *dst);
extern "C" __global__ AICORE void TEXPANDS_f32_64x64_scalar2(__gm__ float *dst);
extern "C" __global__ AICORE void TEXPANDS_f32_16x64_partial(__gm__ float *dst);
extern "C" __global__ AICORE void TEXPANDS_f32_64x64_valid_60x60(__gm__ float *dst);

void LaunchTEXPANDS_f32_16x64_scalar5(float *dst, void *stream) {
    TEXPANDS_f32_16x64_scalar5<<<1, nullptr, stream>>>((__gm__ float *)dst);
}

void LaunchTEXPANDS_f32_32x32_scalar3(float *dst, void *stream) {
    TEXPANDS_f32_32x32_scalar3<<<1, nullptr, stream>>>((__gm__ float *)dst);
}

void LaunchTEXPANDS_f32_64x64_scalar2(float *dst, void *stream) {
    TEXPANDS_f32_64x64_scalar2<<<1, nullptr, stream>>>((__gm__ float *)dst);
}

void LaunchTEXPANDS_f32_16x64_partial(float *dst, void *stream) {
    TEXPANDS_f32_16x64_partial<<<1, nullptr, stream>>>((__gm__ float *)dst);
}

void LaunchTEXPANDS_f32_64x64_valid_60x60(float *dst, void *stream) {
    TEXPANDS_f32_64x64_valid_60x60<<<1, nullptr, stream>>>((__gm__ float *)dst);
}

// ========== int32 kernels ==========

extern "C" __global__ AICORE void TEXPANDS_i32_64x64_scalar100(__gm__ int32_t *dst);
extern "C" __global__ AICORE void TEXPANDS_i32_64x64_valid_60x60(__gm__ int32_t *dst);

void LaunchTEXPANDS_i32_64x64_scalar100(int32_t *dst, void *stream) {
    TEXPANDS_i32_64x64_scalar100<<<1, nullptr, stream>>>((__gm__ int32_t *)dst);
}

void LaunchTEXPANDS_i32_64x64_valid_60x60(int32_t *dst, void *stream) {
    TEXPANDS_i32_64x64_valid_60x60<<<1, nullptr, stream>>>((__gm__ int32_t *)dst);
}

// ========== half (fp16) kernels ==========

extern "C" __global__ AICORE void TEXPANDS_f16_64x64_scalar1_5(__gm__ uint16_t *dst);
extern "C" __global__ AICORE void TEXPANDS_f16_2x4096_valid_1x3600(__gm__ uint16_t *dst);

void LaunchTEXPANDS_f16_64x64_scalar1_5(uint16_t *dst, void *stream) {
    TEXPANDS_f16_64x64_scalar1_5<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst);
}

void LaunchTEXPANDS_f16_2x4096_valid_1x3600(uint16_t *dst, void *stream) {
    TEXPANDS_f16_2x4096_valid_1x3600<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst);
}

// ========== int16 kernels ==========

extern "C" __global__ AICORE void TEXPANDS_i16_64x64_scalar50(__gm__ int16_t *dst);
extern "C" __global__ AICORE void TEXPANDS_i16_20x512_valid_16x200(__gm__ int16_t *dst);

void LaunchTEXPANDS_i16_64x64_scalar50(int16_t *dst, void *stream) {
    TEXPANDS_i16_64x64_scalar50<<<1, nullptr, stream>>>((__gm__ int16_t *)dst);
}

void LaunchTEXPANDS_i16_20x512_valid_16x200(int16_t *dst, void *stream) {
    TEXPANDS_i16_20x512_valid_16x200<<<1, nullptr, stream>>>((__gm__ int16_t *)dst);
}