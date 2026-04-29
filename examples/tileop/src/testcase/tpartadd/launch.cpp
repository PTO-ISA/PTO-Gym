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

// Case 0: f32 64x64 full
extern "C" __global__ AICORE void TPARTADD_f32_64x64_full(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTADD_f32_64x64_full(float *a, float *b, float *c, void *stream) {
    TPARTADD_f32_64x64_full<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 1: f32 64x64 src0 row less
extern "C" __global__ AICORE void TPARTADD_f32_64x64_src0_row_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTADD_f32_64x64_src0_row_less(float *a, float *b, float *c, void *stream) {
    TPARTADD_f32_64x64_src0_row_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 2: f32 64x64 src0 col less
extern "C" __global__ AICORE void TPARTADD_f32_64x64_src0_col_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTADD_f32_64x64_src0_col_less(float *a, float *b, float *c, void *stream) {
    TPARTADD_f32_64x64_src0_col_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 3: f32 64x64 src1 row less
extern "C" __global__ AICORE void TPARTADD_f32_64x64_src1_row_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTADD_f32_64x64_src1_row_less(float *a, float *b, float *c, void *stream) {
    TPARTADD_f32_64x64_src1_row_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 4: f32 64x64 src1 col less
extern "C" __global__ AICORE void TPARTADD_f32_64x64_src1_col_less(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTPARTADD_f32_64x64_src1_col_less(float *a, float *b, float *c, void *stream) {
    TPARTADD_f32_64x64_src1_col_less<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 5: f16 8x48 src0 col less
extern "C" __global__ AICORE void TPARTADD_f16_8x48_src0_col_less(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);

void LaunchTPARTADD_f16_8x48_src0_col_less(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TPARTADD_f16_8x48_src0_col_less<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

// Case 6: f16 8x768 src0 col less
extern "C" __global__ AICORE void TPARTADD_f16_8x768_src0_col_less(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);

void LaunchTPARTADD_f16_8x768_src0_col_less(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TPARTADD_f16_8x768_src0_col_less<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

// Case 7: i16 8x48 src1 col less
extern "C" __global__ AICORE void TPARTADD_i16_8x48_src1_col_less(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTPARTADD_i16_8x48_src1_col_less(int16_t *a, int16_t *b, int16_t *c, void *stream) {
    TPARTADD_i16_8x48_src1_col_less<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}

// Case 8: i32 64x64 src0 row less
extern "C" __global__ AICORE void TPARTADD_i32_64x64_src0_row_less(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTPARTADD_i32_64x64_src0_row_less(int32_t *a, int32_t *b, int32_t *c, void *stream) {
    TPARTADD_i32_64x64_src0_row_less<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}