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

extern "C" __global__ AICORE void TCVT_f32_to_i32_rint_16x64(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_round_16x64(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_rint_16x64(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_rint_16x64(__gm__ float *src, __gm__ half *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_rint_16x64(__gm__ half *src, __gm__ float *dst);

void LaunchTCVT_f32_to_i32_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_rint_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_round_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_round_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i32_to_f32_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_rint_16x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f16_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_rint_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ half *)dst);
}

void LaunchTCVT_f16_to_f32_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_rint_16x64<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ float *)dst);
}
