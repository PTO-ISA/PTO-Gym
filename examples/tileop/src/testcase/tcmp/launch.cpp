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

// Case 1: f16 32x32 eq (half_32x32_32x32)
extern "C" __global__ AICORE void TCMP_f16_32x32_eq(__gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ int8_t *c);

void LaunchTCMP_f16_32x32_eq(uint16_t *a, uint16_t *b, int8_t *c, void *stream) {
    TCMP_f16_32x32_eq<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ int8_t *)c);
}

// Case 2: f32 8x64 gt (float_8x64_8x64)
extern "C" __global__ AICORE void TCMP_f32_8x64_gt(__gm__ float *a, __gm__ float *b, __gm__ int8_t *c);

void LaunchTCMP_f32_8x64_gt(float *a, float *b, int8_t *c, void *stream) {
    TCMP_f32_8x64_gt<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ int8_t *)c);
}

// Case 3: i32 4x64 ne (int32_4x64_4x64)
extern "C" __global__ AICORE void TCMP_i32_4x64_ne(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int8_t *c);

void LaunchTCMP_i32_4x64_ne(int32_t *a, int32_t *b, int8_t *c, void *stream) {
    TCMP_i32_4x64_ne<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int8_t *)c);
}

// Case 4: i32 128x128 lt with valid 64x64 (int32_128x128_64x64)
extern "C" __global__ AICORE void TCMP_i32_128x128_lt(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int8_t *c);

void LaunchTCMP_i32_128x128_lt(int32_t *a, int32_t *b, int8_t *c, void *stream) {
    TCMP_i32_128x128_lt<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int8_t *)c);
}

// Case 5: i32 64x64 eq with valid 32x32 (int32_64x64_32x32)
extern "C" __global__ AICORE void TCMP_i32_64x64_eq(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int8_t *c);

void LaunchTCMP_i32_64x64_eq(int32_t *a, int32_t *b, int8_t *c, void *stream) {
    TCMP_i32_64x64_eq<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int8_t *)c);
}

// Case 6: i32 16x32 eq (int32_16x32_16x32)
extern "C" __global__ AICORE void TCMP_i32_16x32_eq(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int8_t *c);

void LaunchTCMP_i32_16x32_eq(int32_t *a, int32_t *b, int8_t *c, void *stream) {
    TCMP_i32_16x32_eq<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int8_t *)c);
}

// Case 7: f32 128x128 le with valid 64x64 (float_128x128_64x64)
extern "C" __global__ AICORE void TCMP_f32_128x128_le(__gm__ float *a, __gm__ float *b, __gm__ int8_t *c);

void LaunchTCMP_f32_128x128_le(float *a, float *b, int8_t *c, void *stream) {
    TCMP_f32_128x128_le<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ int8_t *)c);
}

// Case 8: i32 77x80 eq with valid 32x32 (int32_77x80_32x32)
extern "C" __global__ AICORE void TCMP_i32_77x80_eq(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int8_t *c);

void LaunchTCMP_i32_77x80_eq(int32_t *a, int32_t *b, int8_t *c, void *stream) {
    TCMP_i32_77x80_eq<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int8_t *)c);
}

// Case 9: i32 32x32 eq (int32_32x32_32x32)
extern "C" __global__ AICORE void TCMP_i32_32x32_eq(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int8_t *c);

void LaunchTCMP_i32_32x32_eq(int32_t *a, int32_t *b, int8_t *c, void *stream) {
    TCMP_i32_32x32_eq<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int8_t *)c);
}

// Case 10: i16 32x32 eq with valid 16x32 (int16_32x32_16x32)
extern "C" __global__ AICORE void TCMP_i16_32x32_eq(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int8_t *c);

void LaunchTCMP_i16_32x32_eq(int16_t *a, int16_t *b, int8_t *c, void *stream) {
    TCMP_i16_32x32_eq<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int8_t *)c);
}

// Case 11: i16 77x80 le with valid 32x32 (int16_77x80_32x32)
extern "C" __global__ AICORE void TCMP_i16_77x80_le(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int8_t *c);

void LaunchTCMP_i16_77x80_le(int16_t *a, int16_t *b, int8_t *c, void *stream) {
    TCMP_i16_77x80_le<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int8_t *)c);
}