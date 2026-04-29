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
extern "C" __global__ AICORE void TROWEXPAND_f32_16x128(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TROWEXPAND_f32_16x127(__gm__ float *src, __gm__ float *dst);

void LaunchTROWEXPAND_f32_16x128(float *src, float *dst, void *stream) {
    TROWEXPAND_f32_16x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}
void LaunchTROWEXPAND_f32_16x127(float *src, float *dst, void *stream) {
    TROWEXPAND_f32_16x127<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// f16 kernels (use uint16_t for aclFloat16)
extern "C" __global__ AICORE void TROWEXPAND_f16_16x512(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TROWEXPAND_f16_16x511(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTROWEXPAND_f16_16x512(void *src, void *dst, void *stream) {
    TROWEXPAND_f16_16x512<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}
void LaunchTROWEXPAND_f16_16x511(void *src, void *dst, void *stream) {
    TROWEXPAND_f16_16x511<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

// i8 kernels
extern "C" __global__ AICORE void TROWEXPAND_i8_16x256(__gm__ int8_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TROWEXPAND_i8_16x255(__gm__ int8_t *src, __gm__ int8_t *dst);

void LaunchTROWEXPAND_i8_16x256(void *src, void *dst, void *stream) {
    TROWEXPAND_i8_16x256<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst);
}
void LaunchTROWEXPAND_i8_16x255(void *src, void *dst, void *stream) {
    TROWEXPAND_i8_16x255<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst);
}