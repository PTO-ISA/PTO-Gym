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

// Case 0: int8 64x64
extern "C" __global__ AICORE void TNOT_int8_64x64(__gm__ int8_t *a, __gm__ int8_t *b);

void LaunchTNOT_int8_64x64(void *a, void *b, void *stream) {
    TNOT_int8_64x64<<<1, nullptr, stream>>>((__gm__ int8_t *)a, (__gm__ int8_t *)b);
}

// Case 1: uint8 60x60
extern "C" __global__ AICORE void TNOT_uint8_60x60(__gm__ uint8_t *a, __gm__ uint8_t *b);

void LaunchTNOT_uint8_60x60(void *a, void *b, void *stream) {
    TNOT_uint8_60x60<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b);
}

// Case 2: int16 64x64
extern "C" __global__ AICORE void TNOT_int16_64x64(__gm__ int16_t *a, __gm__ int16_t *b);

void LaunchTNOT_int16_64x64(void *a, void *b, void *stream) {
    TNOT_int16_64x64<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b);
}

// Case 3: uint16 60x60
extern "C" __global__ AICORE void TNOT_uint16_60x60(__gm__ uint16_t *a, __gm__ uint16_t *b);

void LaunchTNOT_uint16_60x60(void *a, void *b, void *stream) {
    TNOT_uint16_60x60<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b);
}

// Case 4: int32 64x64
extern "C" __global__ AICORE void TNOT_int32_64x64(__gm__ int32_t *a, __gm__ int32_t *b);

void LaunchTNOT_int32_64x64(void *a, void *b, void *stream) {
    TNOT_int32_64x64<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b);
}

// Case 5: uint32 60x60
extern "C" __global__ AICORE void TNOT_uint32_60x60(__gm__ uint32_t *a, __gm__ uint32_t *b);

void LaunchTNOT_uint32_60x60(void *a, void *b, void *stream) {
    TNOT_uint32_60x60<<<1, nullptr, stream>>>((__gm__ uint32_t *)a, (__gm__ uint32_t *)b);
}