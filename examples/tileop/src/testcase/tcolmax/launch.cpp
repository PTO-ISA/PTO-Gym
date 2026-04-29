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

// Case 0: f32 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_f32_1x256(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLMAX_f32_1x256(float *dst, float *src, void *stream) {
    TCOLMAX_f32_1x256<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 1: f32 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_f32_16x128(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLMAX_f32_16x128(float *dst, float *src, void *stream) {
    TCOLMAX_f32_16x128<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 2: f32 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_f32_16x256(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLMAX_f32_16x256(float *dst, float *src, void *stream) {
    TCOLMAX_f32_16x256<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 3: f16 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_f16_1x256(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLMAX_f16_1x256(void *dst, void *src, void *stream) {
    TCOLMAX_f16_1x256<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 4: f16 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_f16_16x128(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLMAX_f16_16x128(void *dst, void *src, void *stream) {
    TCOLMAX_f16_16x128<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 5: f16 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_f16_16x256(__gm__ half *dst, __gm__ half *src);

void LaunchTCOLMAX_f16_16x256(void *dst, void *src, void *stream) {
    TCOLMAX_f16_16x256<<<1, nullptr, stream>>>((__gm__ half *)dst, (__gm__ half *)src);
}

// Case 6: i8 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_i8_1x256(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLMAX_i8_1x256(void *dst, void *src, void *stream) {
    TCOLMAX_i8_1x256<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}

// Case 7: i8 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_i8_16x128(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLMAX_i8_16x128(void *dst, void *src, void *stream) {
    TCOLMAX_i8_16x128<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}

// Case 8: i8 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_i8_16x256(__gm__ int8_t *dst, __gm__ int8_t *src);

void LaunchTCOLMAX_i8_16x256(void *dst, void *src, void *stream) {
    TCOLMAX_i8_16x256<<<1, nullptr, stream>>>((__gm__ int8_t *)dst, (__gm__ int8_t *)src);
}

// Case 9: i16 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_i16_1x256(__gm__ int16_t *dst, __gm__ int16_t *src);

void LaunchTCOLMAX_i16_1x256(void *dst, void *src, void *stream) {
    TCOLMAX_i16_1x256<<<1, nullptr, stream>>>((__gm__ int16_t *)dst, (__gm__ int16_t *)src);
}

// Case 10: i16 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_i16_16x128(__gm__ int16_t *dst, __gm__ int16_t *src);

void LaunchTCOLMAX_i16_16x128(void *dst, void *src, void *stream) {
    TCOLMAX_i16_16x128<<<1, nullptr, stream>>>((__gm__ int16_t *)dst, (__gm__ int16_t *)src);
}

// Case 11: i16 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_i16_16x256(__gm__ int16_t *dst, __gm__ int16_t *src);

void LaunchTCOLMAX_i16_16x256(void *dst, void *src, void *stream) {
    TCOLMAX_i16_16x256<<<1, nullptr, stream>>>((__gm__ int16_t *)dst, (__gm__ int16_t *)src);
}

// Case 12: i32 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_i32_1x256(__gm__ int32_t *dst, __gm__ int32_t *src);

void LaunchTCOLMAX_i32_1x256(void *dst, void *src, void *stream) {
    TCOLMAX_i32_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int32_t *)src);
}

// Case 13: i32 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_i32_16x128(__gm__ int32_t *dst, __gm__ int32_t *src);

void LaunchTCOLMAX_i32_16x128(void *dst, void *src, void *stream) {
    TCOLMAX_i32_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int32_t *)src);
}

// Case 14: i32 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_i32_16x256(__gm__ int32_t *dst, __gm__ int32_t *src);

void LaunchTCOLMAX_i32_16x256(void *dst, void *src, void *stream) {
    TCOLMAX_i32_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int32_t *)src);
}

// Case 15: ui8 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_ui8_1x256(__gm__ uint8_t *dst, __gm__ uint8_t *src);

void LaunchTCOLMAX_ui8_1x256(void *dst, void *src, void *stream) {
    TCOLMAX_ui8_1x256<<<1, nullptr, stream>>>((__gm__ uint8_t *)dst, (__gm__ uint8_t *)src);
}

// Case 16: ui8 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_ui8_16x128(__gm__ uint8_t *dst, __gm__ uint8_t *src);

void LaunchTCOLMAX_ui8_16x128(void *dst, void *src, void *stream) {
    TCOLMAX_ui8_16x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)dst, (__gm__ uint8_t *)src);
}

// Case 17: ui8 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_ui8_16x256(__gm__ uint8_t *dst, __gm__ uint8_t *src);

void LaunchTCOLMAX_ui8_16x256(void *dst, void *src, void *stream) {
    TCOLMAX_ui8_16x256<<<1, nullptr, stream>>>((__gm__ uint8_t *)dst, (__gm__ uint8_t *)src);
}

// Case 18: ui16 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_ui16_1x256(__gm__ uint16_t *dst, __gm__ uint16_t *src);

void LaunchTCOLMAX_ui16_1x256(void *dst, void *src, void *stream) {
    TCOLMAX_ui16_1x256<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst, (__gm__ uint16_t *)src);
}

// Case 19: ui16 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_ui16_16x128(__gm__ uint16_t *dst, __gm__ uint16_t *src);

void LaunchTCOLMAX_ui16_16x128(void *dst, void *src, void *stream) {
    TCOLMAX_ui16_16x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst, (__gm__ uint16_t *)src);
}

// Case 20: ui16 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_ui16_16x256(__gm__ uint16_t *dst, __gm__ uint16_t *src);

void LaunchTCOLMAX_ui16_16x256(void *dst, void *src, void *stream) {
    TCOLMAX_ui16_16x256<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst, (__gm__ uint16_t *)src);
}

// Case 21: ui32 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_ui32_1x256(__gm__ uint32_t *dst, __gm__ uint32_t *src);

void LaunchTCOLMAX_ui32_1x256(void *dst, void *src, void *stream) {
    TCOLMAX_ui32_1x256<<<1, nullptr, stream>>>((__gm__ uint32_t *)dst, (__gm__ uint32_t *)src);
}

// Case 22: ui32 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLMAX_ui32_16x128(__gm__ uint32_t *dst, __gm__ uint32_t *src);

void LaunchTCOLMAX_ui32_16x128(void *dst, void *src, void *stream) {
    TCOLMAX_ui32_16x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)dst, (__gm__ uint32_t *)src);
}

// Case 23: ui32 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLMAX_ui32_16x256(__gm__ uint32_t *dst, __gm__ uint32_t *src);

void LaunchTCOLMAX_ui32_16x256(void *dst, void *src, void *stream) {
    TCOLMAX_ui32_16x256<<<1, nullptr, stream>>>((__gm__ uint32_t *)dst, (__gm__ uint32_t *)src);
}