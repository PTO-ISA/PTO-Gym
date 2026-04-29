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

// Case: f32 64x64 full
extern "C" __global__ AICORE void TPARTMAX_f32_64x64_full(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMAX_f32_64x64_full(float *a, float *b, float *c, void *stream) {
    TPARTMAX_f32_64x64_full<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case: f32 2x24 src1 col less
extern "C" __global__ AICORE void TPARTMAX_f32_2x24_src1_col_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMAX_f32_2x24_src1_col_less(float *a, float *b, float *c, void *stream) {
    TPARTMAX_f32_2x24_src1_col_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case: f32 128x64 src1 row less
extern "C" __global__ AICORE void TPARTMAX_f32_128x64_src1_row_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMAX_f32_128x64_src1_row_less(float *a, float *b, float *c, void *stream) {
    TPARTMAX_f32_128x64_src1_row_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case: f32 95x95 full
extern "C" __global__ AICORE void TPARTMAX_f32_95x95_full(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMAX_f32_95x95_full(float *a, float *b, float *c, void *stream) {
    TPARTMAX_f32_95x95_full<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case: f32 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_f32_122x123_complex(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTMAX_f32_122x123_complex(float *a, float *b, float *c, void *stream) {
    TPARTMAX_f32_122x123_complex<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case: f16 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_f16_122x123_complex(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);

void LaunchTPARTMAX_f16_122x123_complex(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TPARTMAX_f16_122x123_complex<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

// Case: i16 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_i16_122x123_complex(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTPARTMAX_i16_122x123_complex(int16_t *a, int16_t *b, int16_t *c, void *stream) {
    TPARTMAX_i16_122x123_complex<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}

// Case: i32 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_i32_122x123_complex(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTPARTMAX_i32_122x123_complex(int32_t *a, int32_t *b, int32_t *c, void *stream) {
    TPARTMAX_i32_122x123_complex<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}

// Case: u16 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_u16_122x123_complex(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);

void LaunchTPARTMAX_u16_122x123_complex(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TPARTMAX_u16_122x123_complex<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

// Case: u32 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_u32_122x123_complex(__gm__ uint32_t *a, __gm__ uint32_t *b, __gm__ uint32_t *c);

void LaunchTPARTMAX_u32_122x123_complex(uint32_t *a, uint32_t *b, uint32_t *c, void *stream) {
    TPARTMAX_u32_122x123_complex<<<1, nullptr, stream>>>((__gm__ uint32_t *)a, (__gm__ uint32_t *)b, (__gm__ uint32_t *)c);
}

// Case: i8 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_i8_122x123_complex(__gm__ int8_t *a, __gm__ int8_t *b, __gm__ int8_t *c);

void LaunchTPARTMAX_i8_122x123_complex(int8_t *a, int8_t *b, int8_t *c, void *stream) {
    TPARTMAX_i8_122x123_complex<<<1, nullptr, stream>>>((__gm__ int8_t *)a, (__gm__ int8_t *)b, (__gm__ int8_t *)c);
}

// Case: u8 122x123 complex
extern "C" __global__ AICORE void TPARTMAX_u8_122x123_complex(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c);

void LaunchTPARTMAX_u8_122x123_complex(uint8_t *a, uint8_t *b, uint8_t *c, void *stream) {
    TPARTMAX_u8_122x123_complex<<<1, nullptr, stream>>>((__gm__ uint8_t *)a, (__gm__ uint8_t *)b, (__gm__ uint8_t *)c);
}