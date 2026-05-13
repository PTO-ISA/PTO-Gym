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

// Case 0: f32 1x256 (input: 1x256, tmp: 1x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_f32_1x256(__gm__ int32_t *dst, __gm__ float *tmp, __gm__ float *src);

void LaunchTCOLARGMAX_f32_1x256(int32_t *dst, float *tmp, float *src, void *stream) {
    TCOLARGMAX_f32_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ float *)tmp, (__gm__ float *)src);
}

// Case 1: f32 16x128 (input: 16x128, tmp: 16x128, output: 1x128 indices)
extern "C" __global__ AICORE void TCOLARGMAX_f32_16x128(__gm__ int32_t *dst, __gm__ float *tmp, __gm__ float *src);

void LaunchTCOLARGMAX_f32_16x128(int32_t *dst, float *tmp, float *src, void *stream) {
    TCOLARGMAX_f32_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ float *)tmp, (__gm__ float *)src);
}

// Case 2: f32 16x256 (input: 16x256, tmp: 16x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_f32_16x256(__gm__ int32_t *dst, __gm__ float *tmp, __gm__ float *src);

void LaunchTCOLARGMAX_f32_16x256(int32_t *dst, float *tmp, float *src, void *stream) {
    TCOLARGMAX_f32_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ float *)tmp, (__gm__ float *)src);
}

// Case 3: f16 1x256 (input: 1x256, tmp: 1x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_f16_1x256(__gm__ int32_t *dst, __gm__ half *tmp, __gm__ half *src);

void LaunchTCOLARGMAX_f16_1x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_f16_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ half *)tmp, (__gm__ half *)src);
}

// Case 4: f16 16x128 (input: 16x128, tmp: 16x128, output: 1x128 indices)
extern "C" __global__ AICORE void TCOLARGMAX_f16_16x128(__gm__ int32_t *dst, __gm__ half *tmp, __gm__ half *src);

void LaunchTCOLARGMAX_f16_16x128(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_f16_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ half *)tmp, (__gm__ half *)src);
}

// Case 5: f16 16x256 (input: 16x256, tmp: 16x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_f16_16x256(__gm__ int32_t *dst, __gm__ half *tmp, __gm__ half *src);

void LaunchTCOLARGMAX_f16_16x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_f16_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ half *)tmp, (__gm__ half *)src);
}

// Case 6: ui32 1x256 (input: 1x256, tmp: 1x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui32_1x256(__gm__ int32_t *dst, __gm__ uint32_t *tmp, __gm__ uint32_t *src);

void LaunchTCOLARGMAX_ui32_1x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui32_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint32_t *)tmp, (__gm__ uint32_t *)src);
}

// Case 7: ui32 16x128 (input: 16x128, tmp: 16x128, output: 1x128 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui32_16x128(__gm__ int32_t *dst, __gm__ uint32_t *tmp, __gm__ uint32_t *src);

void LaunchTCOLARGMAX_ui32_16x128(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui32_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint32_t *)tmp, (__gm__ uint32_t *)src);
}

// Case 8: ui32 16x256 (input: 16x256, tmp: 16x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui32_16x256(__gm__ int32_t *dst, __gm__ uint32_t *tmp, __gm__ uint32_t *src);

void LaunchTCOLARGMAX_ui32_16x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui32_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint32_t *)tmp, (__gm__ uint32_t *)src);
}

// Case 9: ui16 1x256 (input: 1x256, tmp: 1x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui16_1x256(__gm__ int32_t *dst, __gm__ uint16_t *tmp, __gm__ uint16_t *src);

void LaunchTCOLARGMAX_ui16_1x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui16_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint16_t *)tmp, (__gm__ uint16_t *)src);
}

// Case 10: ui16 16x128 (input: 16x128, tmp: 16x128, output: 1x128 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui16_16x128(__gm__ int32_t *dst, __gm__ uint16_t *tmp, __gm__ uint16_t *src);

void LaunchTCOLARGMAX_ui16_16x128(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui16_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint16_t *)tmp, (__gm__ uint16_t *)src);
}

// Case 11: ui16 16x256 (input: 16x256, tmp: 16x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui16_16x256(__gm__ int32_t *dst, __gm__ uint16_t *tmp, __gm__ uint16_t *src);

void LaunchTCOLARGMAX_ui16_16x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui16_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint16_t *)tmp, (__gm__ uint16_t *)src);
}

// Case 12: ui8 1x256 (input: 1x256, tmp: 1x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui8_1x256(__gm__ int32_t *dst, __gm__ uint8_t *tmp, __gm__ uint8_t *src);

void LaunchTCOLARGMAX_ui8_1x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui8_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint8_t *)tmp, (__gm__ uint8_t *)src);
}

// Case 13: ui8 16x128 (input: 16x128, tmp: 16x128, output: 1x128 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui8_16x128(__gm__ int32_t *dst, __gm__ uint8_t *tmp, __gm__ uint8_t *src);

void LaunchTCOLARGMAX_ui8_16x128(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui8_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint8_t *)tmp, (__gm__ uint8_t *)src);
}

// Case 14: ui8 16x256 (input: 16x256, tmp: 16x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_ui8_16x256(__gm__ int32_t *dst, __gm__ uint8_t *tmp, __gm__ uint8_t *src);

void LaunchTCOLARGMAX_ui8_16x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_ui8_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ uint8_t *)tmp, (__gm__ uint8_t *)src);
}

// Case 15: i8 1x256 (input: 1x256, tmp: 1x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_i8_1x256(__gm__ int32_t *dst, __gm__ int8_t *tmp, __gm__ int8_t *src);

void LaunchTCOLARGMAX_i8_1x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_i8_1x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int8_t *)tmp, (__gm__ int8_t *)src);
}

// Case 16: i8 16x128 (input: 16x128, tmp: 16x128, output: 1x128 indices)
extern "C" __global__ AICORE void TCOLARGMAX_i8_16x128(__gm__ int32_t *dst, __gm__ int8_t *tmp, __gm__ int8_t *src);

void LaunchTCOLARGMAX_i8_16x128(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_i8_16x128<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int8_t *)tmp, (__gm__ int8_t *)src);
}

// Case 17: i8 16x256 (input: 16x256, tmp: 16x256, output: 1x256 indices)
extern "C" __global__ AICORE void TCOLARGMAX_i8_16x256(__gm__ int32_t *dst, __gm__ int8_t *tmp, __gm__ int8_t *src);

void LaunchTCOLARGMAX_i8_16x256(void *dst, void *tmp, void *src, void *stream) {
    TCOLARGMAX_i8_16x256<<<1, nullptr, stream>>>((__gm__ int32_t *)dst, (__gm__ int8_t *)tmp, (__gm__ int8_t *)src);
}