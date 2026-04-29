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

// ========== Case 1: float, 128x128, valid=128x127 ==========

extern "C" __global__ AICORE void TFILLPAD_f32_128x128_pad_128x127(__gm__ float *src, __gm__ float *dst);

void LaunchTFILLPAD_f32_128x128_pad_128x127(float *src, float *dst, void *stream) {
    TFILLPAD_f32_128x128_pad_128x127<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// ========== Case 2: float, 128x160, valid=128x127 ==========

extern "C" __global__ AICORE void TFILLPAD_f32_128x160_pad_128x127(__gm__ float *src, __gm__ float *dst);

void LaunchTFILLPAD_f32_128x160_pad_128x127(float *src, float *dst, void *stream) {
    TFILLPAD_f32_128x160_pad_128x127<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// ========== Case 3: float, 128x160, valid=128x127 (different PadVal) ==========

extern "C" __global__ AICORE void TFILLPAD_f32_128x160_pad_128x127_v2(__gm__ float *src, __gm__ float *dst);

void LaunchTFILLPAD_f32_128x160_pad_128x127_v2(float *src, float *dst, void *stream) {
    TFILLPAD_f32_128x160_pad_128x127_v2<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// ========== Case 4: float, 260x16, valid=260x7 ==========

extern "C" __global__ AICORE void TFILLPAD_f32_260x16_pad_260x7(__gm__ float *src, __gm__ float *dst);

void LaunchTFILLPAD_f32_260x16_pad_260x7(float *src, float *dst, void *stream) {
    TFILLPAD_f32_260x16_pad_260x7<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// ========== Case 6: uint16, 260x32, valid=260x7 ==========

extern "C" __global__ AICORE void TFILLPAD_u16_260x32_pad_260x7(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTFILLPAD_u16_260x32_pad_260x7(uint16_t *src, uint16_t *dst, void *stream) {
    TFILLPAD_u16_260x32_pad_260x7<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

// ========== Case 7: int8, 260x64, valid=260x7 ==========

extern "C" __global__ AICORE void TFILLPAD_s8_260x64_pad_260x7(__gm__ int8_t *src, __gm__ int8_t *dst);

void LaunchTFILLPAD_s8_260x64_pad_260x7(int8_t *src, int8_t *dst, void *stream) {
    TFILLPAD_s8_260x64_pad_260x7<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst);
}

// ========== Case 10: int16, 260x32, valid=260x7 ==========

extern "C" __global__ AICORE void TFILLPAD_s16_260x32_pad_260x7(__gm__ int16_t *src, __gm__ int16_t *dst);

void LaunchTFILLPAD_s16_260x32_pad_260x7(int16_t *src, int16_t *dst, void *stream) {
    TFILLPAD_s16_260x32_pad_260x7<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int16_t *)dst);
}

// ========== Case 11: int32, 260x32, valid=260x7 ==========

extern "C" __global__ AICORE void TFILLPAD_s32_260x32_pad_260x7(__gm__ int32_t *src, __gm__ int32_t *dst);

void LaunchTFILLPAD_s32_260x32_pad_260x7(int32_t *src, int32_t *dst, void *stream) {
    TFILLPAD_s32_260x32_pad_260x7<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}

// ========== Case 12: float, src=128x64, dst=128x128, PadCustomNeg1 ==========

extern "C" __global__ AICORE void TFILLPAD_f32_128x128_pad_128x64_neg1(__gm__ float *src, __gm__ float *dst);

void LaunchTFILLPAD_f32_128x128_pad_128x64_neg1(float *src, float *dst, void *stream) {
    TFILLPAD_f32_128x128_pad_128x64_neg1<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// ========== Case 13: float, src=128x127, dst=128x160, PadCustomNeg1 ==========

extern "C" __global__ AICORE void TFILLPAD_f32_128x160_pad_128x127_neg1(__gm__ float *src, __gm__ float *dst);

void LaunchTFILLPAD_f32_128x160_pad_128x127_neg1(float *src, float *dst, void *stream) {
    TFILLPAD_f32_128x160_pad_128x127_neg1<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}