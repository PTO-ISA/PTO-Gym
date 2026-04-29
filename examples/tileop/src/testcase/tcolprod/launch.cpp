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
extern "C" __global__ AICORE void TCOLPROD_f32_1x256(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLPROD_f32_1x256(float *dst, float *src, void *stream) {
    TCOLPROD_f32_1x256<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 1: f32 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLPROD_f32_16x128(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLPROD_f32_16x128(float *dst, float *src, void *stream) {
    TCOLPROD_f32_16x128<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 2: f32 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_f32_16x256(__gm__ float *dst, __gm__ float *src);

void LaunchTCOLPROD_f32_16x256(float *dst, float *src, void *stream) {
    TCOLPROD_f32_16x256<<<1, nullptr, stream>>>((__gm__ float *)dst, (__gm__ float *)src);
}

// Case 3: i16 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_i16_1x256(__gm__ int16_t *dst, __gm__ int16_t *src);

void LaunchTCOLPROD_i16_1x256(void *dst, void *src, void *stream) {
    TCOLPROD_i16_1x256<<<1, nullptr, stream>>>((__gm__ int16_t *)dst, (__gm__ int16_t *)src);
}

// Case 4: i16 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLPROD_i16_16x128(__gm__ int16_t *dst, __gm__ int16_t *src);

void LaunchTCOLPROD_i16_16x128(void *dst, void *src, void *stream) {
    TCOLPROD_i16_16x128<<<1, nullptr, stream>>>((__gm__ int16_t *)dst, (__gm__ int16_t *)src);
}

// Case 5: i16 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_i16_16x256(__gm__ int16_t *dst, __gm__ int16_t *src);

void LaunchTCOLPROD_i16_16x256(void *dst, void *src, void *stream) {
    TCOLPROD_i16_16x256<<<1, nullptr, stream>>>((__gm__ int16_t *)dst, (__gm__ int16_t *)src);
}

// Case 6: ui16 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_ui16_1x256(__gm__ uint16_t *dst, __gm__ uint16_t *src);

void LaunchTCOLPROD_ui16_1x256(void *dst, void *src, void *stream) {
    TCOLPROD_ui16_1x256<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst, (__gm__ uint16_t *)src);
}

// Case 7: ui16 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLPROD_ui16_16x128(__gm__ uint16_t *dst, __gm__ uint16_t *src);

void LaunchTCOLPROD_ui16_16x128(void *dst, void *src, void *stream) {
    TCOLPROD_ui16_16x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst, (__gm__ uint16_t *)src);
}

// Case 8: ui16 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_ui16_16x256(__gm__ uint16_t *dst, __gm__ uint16_t *src);

void LaunchTCOLPROD_ui16_16x256(void *dst, void *src, void *stream) {
    TCOLPROD_ui16_16x256<<<1, nullptr, stream>>>((__gm__ uint16_t *)dst, (__gm__ uint16_t *)src);
}

// Case 9: i32 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_i32_1x256(__gm__ int32_t *dst, __gm__ int32_t *src);

void LaunchTCOLPROD_i32_1x256(void *dst, void *src, void *stream) {
    TCOLPROD_i32_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int32_t *)src);
}

// Case 10: i32 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLPROD_i32_16x128(__gm__ int32_t *dst, __gm__ int32_t *src);

void LaunchTCOLPROD_i32_16x128(void *dst, void *src, void *stream) {
    TCOLPROD_i32_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int32_t *)src);
}

// Case 11: i32 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_i32_16x256(__gm__ int32_t *dst, __gm__ int32_t *src);

void LaunchTCOLPROD_i32_16x256(void *dst, void *src, void *stream) {
    TCOLPROD_i32_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int32_t *)src);
}

// Case 12: ui32 1x256 (input: 1x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_ui32_1x256(__gm__ uint32_t *dst, __gm__ uint32_t *src);

void LaunchTCOLPROD_ui32_1x256(void *dst, void *src, void *stream) {
    TCOLPROD_ui32_1x256<<<1, nullptr, stream>>>((__gm__ uint32_t *)dst, (__gm__ uint32_t *)src);
}

// Case 13: ui32 16x128 (input: 16x128, output: 1x128)
extern "C" __global__ AICORE void TCOLPROD_ui32_16x128(__gm__ uint32_t *dst, __gm__ uint32_t *src);

void LaunchTCOLPROD_ui32_16x128(void *dst, void *src, void *stream) {
    TCOLPROD_ui32_16x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)dst, (__gm__ uint32_t *)src);
}

// Case 14: ui32 16x256 (input: 16x256, output: 1x256)
extern "C" __global__ AICORE void TCOLPROD_ui32_16x256(__gm__ uint32_t *dst, __gm__ uint32_t *src);

void LaunchTCOLPROD_ui32_16x256(void *dst, void *src, void *stream) {
    TCOLPROD_ui32_16x256<<<1, nullptr, stream>>>((__gm__ uint32_t *)dst, (__gm__ uint32_t *)src);
}