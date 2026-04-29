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

// ========== uint16 kernel (C++ case 8) ==========

extern "C" __global__ AICORE void TFILLPAD_EXPAND_u16_260x32_src_259x7(__gm__ uint16_t *src, __gm__ uint16_t *dst);

void LaunchTFILLPAD_EXPAND_u16_260x32_src_259x7(uint16_t *src, uint16_t *dst, void *stream) {
    TFILLPAD_EXPAND_u16_260x32_src_259x7<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

// ========== int8 kernel (C++ case 9) ==========

extern "C" __global__ AICORE void TFILLPAD_EXPAND_s8_260x64_src_259x7(__gm__ int8_t *src, __gm__ int8_t *dst);

void LaunchTFILLPAD_EXPAND_s8_260x64_src_259x7(int8_t *src, int8_t *dst, void *stream) {
    TFILLPAD_EXPAND_s8_260x64_src_259x7<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int8_t *)dst);
}