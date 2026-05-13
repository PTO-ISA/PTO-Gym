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

// ========== Case: float, 260x16, no expansion (inplace single buffer) ==========

extern "C" __global__ AICORE void TFILLPAD_INPLACE_f32_260x16_noexpand(__gm__ float *buf);

void LaunchTFILLPAD_INPLACE_f32_260x16_noexpand(float *buf, float *dummy, void *stream) {
    // Inplace kernel: single buffer, src == dst physically
    // dummy parameter ignored, only buf is used
    TFILLPAD_INPLACE_f32_260x16_noexpand<<<1, nullptr, stream>>>((__gm__ float *)buf);
}