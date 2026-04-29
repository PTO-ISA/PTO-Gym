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

// Case 1: fp32_32_16_1_16
extern "C" __global__ AICORE void TCOLEXPANDEXPDIF_fp32_32_16_1_16(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTCOLEXPANDEXPDIF_fp32_32_16_1_16(float *src0, float *src1, float *dst, void *stream) {
    TCOLEXPANDEXPDIF_fp32_32_16_1_16<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 2: fp32_16_32_1_32
extern "C" __global__ AICORE void TCOLEXPANDEXPDIF_fp32_16_32_1_32(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTCOLEXPANDEXPDIF_fp32_16_32_1_32(float *src0, float *src1, float *dst, void *stream) {
    TCOLEXPANDEXPDIF_fp32_16_32_1_32<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

// Case 3: fp16_32_32_1_32
extern "C" __global__ AICORE void TCOLEXPANDEXPDIF_fp16_32_32_1_32(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTCOLEXPANDEXPDIF_fp16_32_32_1_32(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TCOLEXPANDEXPDIF_fp16_32_32_1_32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}

// Case 4: fp16_16_128_1_128
extern "C" __global__ AICORE void TCOLEXPANDEXPDIF_fp16_16_128_1_128(__gm__ uint16_t *src0, __gm__ uint16_t *src1, __gm__ uint16_t *dst);

void LaunchTCOLEXPANDEXPDIF_fp16_16_128_1_128(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TCOLEXPANDEXPDIF_fp16_16_128_1_128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src0, (__gm__ uint16_t *)src1, (__gm__ uint16_t *)dst);
}