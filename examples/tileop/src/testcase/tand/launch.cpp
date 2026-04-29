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

// Case 0: i32 16x64
extern "C" __global__ AICORE void TAND_i32_16x64(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTAND_i32_16x64(int32_t *a, int32_t *b, int32_t *c, void *stream) {
    TAND_i32_16x64<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}

// Case 1: i32 32x32
extern "C" __global__ AICORE void TAND_i32_32x32(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTAND_i32_32x32(int32_t *a, int32_t *b, int32_t *c, void *stream) {
    TAND_i32_32x32<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}