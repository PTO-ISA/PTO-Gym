// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void fixpipe_relu_scalar_vector_f16_cv_kernel(
    __gm__ __fp16 *lhs, __gm__ __fp16 *rhs, __gm__ uint32_t *relu_fp,
    __gm__ __fp16 *out_ub_scalar, __gm__ __fp16 *out_ub_vector,
    __gm__ __fp16 *out_gm_scalar, __gm__ __fp16 *out_gm_vector,
    __gm__ __fp16 *out_l1_scalar, __gm__ __fp16 *out_l1_vector) {
  (void)lhs;
  (void)rhs;
  (void)relu_fp;
  (void)out_ub_scalar;
  (void)out_ub_vector;
  (void)out_gm_scalar;
  (void)out_gm_vector;
  (void)out_l1_scalar;
  (void)out_l1_vector;
}
