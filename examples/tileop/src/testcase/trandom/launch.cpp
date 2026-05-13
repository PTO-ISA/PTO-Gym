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

// Case 0: ui32 4x256
extern "C" __global__ AICORE void TRANDOM_int32_4x256(__gm__ uint32_t *key, __gm__ uint32_t *counter, __gm__ uint32_t *output);

void LaunchTRANDOM_int32_4x256(uint32_t *key, uint32_t *counter, uint32_t *output, void *stream) {
    TRANDOM_int32_4x256<<<1, nullptr, stream>>>((__gm__ uint32_t *)key, (__gm__ uint32_t *)counter, (__gm__ uint32_t *)output);
}