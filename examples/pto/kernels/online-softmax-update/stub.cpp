// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/online-softmax-update
// family: kernels
// target_ops: pto.dma_load, pto.dma_store, pto.vlds, pto.vcmax, pto.vdup, pto.vmax, pto.vexpdif, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv, pto.vsts
// scenarios: online-softmax-update, dynamic-rows-and-seq, max-seq-128, block-rows-8, oldmax-oldsum-qk-to-newmax-newsum-expmax-out
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void online_softmax_update_kernel_2d(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3,
    __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, int32_t v8, int32_t v9) {
  (void)v1; (void)v2; (void)v3; (void)v4;
  (void)v5; (void)v6; (void)v7; (void)v8; (void)v9;
}
