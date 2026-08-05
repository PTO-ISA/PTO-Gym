// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#include <cstdint>
#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void
vmi_dynamic_create_group_mask_s32_reduce_store_kernel(__gm__ float *src, __gm__ float *copy,
                                         __gm__ float *sum, int activeCols);

void LaunchVmi_dynamic_create_group_mask_s32_reduce_store_kernel(float *src, float *copy,
                                                    float *sum, int activeCols,
                                                    void *stream) {
  vmi_dynamic_create_group_mask_s32_reduce_store_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)src, (__gm__ float *)copy, (__gm__ float *)sum,
      activeCols);
}
