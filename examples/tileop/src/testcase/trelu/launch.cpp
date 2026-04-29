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

// Case 0: int32 64x64
extern "C" __global__ AICORE void TRELU_int32_64x64(__gm__ int32_t *input, __gm__ int32_t *output);

void LaunchTRELU_int32_64x64(int32_t *input, int32_t *output, void *stream) {
    TRELU_int32_64x64<<<1, nullptr, stream>>>((__gm__ int32_t *)input, (__gm__ int32_t *)output);
}

// Case 1: f16 64x64 valid 60x60
extern "C" __global__ AICORE void TRELU_f16_64x64_v60x60(__gm__ uint16_t *input, __gm__ uint16_t *output);

void LaunchTRELU_f16_64x64_v60x60(uint16_t *input, uint16_t *output, void *stream) {
    TRELU_f16_64x64_v60x60<<<1, nullptr, stream>>>((__gm__ uint16_t *)input, (__gm__ uint16_t *)output);
}

// Case 2: f32 64x64 valid 60x60
extern "C" __global__ AICORE void TRELU_f32_64x64_v60x60(__gm__ float *input, __gm__ float *output);

void LaunchTRELU_f32_64x64_v60x60(float *input, float *output, void *stream) {
    TRELU_f32_64x64_v60x60<<<1, nullptr, stream>>>((__gm__ float *)input, (__gm__ float *)output);
}