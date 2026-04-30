// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vxor-mask-edge
// family: binary-vector
// target_ops: pto.vxor
// scenarios: core-i16-unsigned, full-mask
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vxor_mask_edge_kernel(__gm__ uint16_t *v1,
                                                        __gm__ uint16_t *v2,
                                                        __gm__ uint16_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
