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

// Case 1: half_1_16_512_512
extern "C" __global__ AICORE void TCOLEXPAND_half_1_16_512_512(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTCOLEXPAND_half_1_16_512_512(uint16_t *src, uint16_t *dst, void *stream) {
    TCOLEXPAND_half_1_16_512_512<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

// Case 2: int8_2_32_256_255
extern "C" __global__ AICORE void TCOLEXPAND_int8_2_32_256_255(__gm__ int8_t *src, __gm__ int8_t *dst);

void LaunchTCOLEXPAND_int8_2_32_256_255(int8_t *src, int8_t *dst, void *stream) {
    TCOLEXPAND_int8_2_32_256_255<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst);
}

// Case 3: float_1_8_128_63
extern "C" __global__ AICORE void TCOLEXPAND_float_1_8_128_63(__gm__ float *src, __gm__ float *dst);

void LaunchTCOLEXPAND_float_1_8_128_63(float *src, float *dst, void *stream) {
    TCOLEXPAND_float_1_8_128_63<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// Case 4: half_1_33_512_512
extern "C" __global__ AICORE void TCOLEXPAND_half_1_33_512_512(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTCOLEXPAND_half_1_33_512_512(uint16_t *src, uint16_t *dst, void *stream) {
    TCOLEXPAND_half_1_33_512_512<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

// Case 5: int8_2_17_256_44
extern "C" __global__ AICORE void TCOLEXPAND_int8_2_17_256_44(__gm__ int8_t *src, __gm__ int8_t *dst);

void LaunchTCOLEXPAND_int8_2_17_256_44(int8_t *src, int8_t *dst, void *stream) {
    TCOLEXPAND_int8_2_17_256_44<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst);
}

// Case 6: float_1_54_64_63
extern "C" __global__ AICORE void TCOLEXPAND_float_1_54_64_63(__gm__ float *src, __gm__ float *dst);

void LaunchTCOLEXPAND_float_1_54_64_63(float *src, float *dst, void *stream) {
    TCOLEXPAND_float_1_54_64_63<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}