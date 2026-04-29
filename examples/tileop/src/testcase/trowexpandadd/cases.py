#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trowexpandadd ST test cases.

trowexpandadd: dst = src0 + broadcast(src1) across columns.
- src1Col determines how src1 is broadcast:
  - src1Col=1: only first column is valid, broadcast to dstCols
  - src1Col=8 (for f32): 8 columns are valid, no broadcast needed
- src1Cols (physical) = 32/sizeof(dtype) for NPU alignment

Template parameters:
  - dstRow, dstCol: dst shape
  - src1Row, src1Col: src1 shape (src1Col is valid columns, not physical)
  - src0eqdst: true means src0 shape equals dst, false means different
"""

import numpy as np

CASES = [
    # launchTRowExpandAdd<float, 16, 32, 16, 1, true>
    {
        "name": "f32_16x32",
        "dtype": np.float32,
        "src0_shape": (16, 32),       # src0eqdst=true, same as dst
        "src0_valid_shape": (16, 32),
        "src1_shape": (16, 8),        # physical: 32/sizeof(f32)=8
        "src1_valid_shape": (16, 1),  # src1Col=1, only first column valid
        "dst_shape": (16, 32),
        "dst_valid_shape": (16, 32),
        "eps": 1e-6,
    },
    # launchTRowExpandAdd<float, 56, 128, 56, 1, true>
    {
        "name": "f32_56x128",
        "dtype": np.float32,
        "src0_shape": (56, 128),      # src0eqdst=true
        "src0_valid_shape": (56, 128),
        "src1_shape": (56, 8),        # physical: 8
        "src1_valid_shape": (56, 1),  # src1Col=1
        "dst_shape": (56, 128),
        "dst_valid_shape": (56, 128),
        "eps": 1e-6,
    },
    # launchTRowExpandAdd<aclFloat16, 48, 64, 48, 1, true>
    {
        "name": "f16_48x64",
        "dtype": np.float16,
        "src0_shape": (48, 64),       # src0eqdst=true
        "src0_valid_shape": (48, 64),
        "src1_shape": (48, 16),       # physical: 32/sizeof(f16)=16
        "src1_valid_shape": (48, 1),  # src1Col=1
        "dst_shape": (48, 64),
        "dst_valid_shape": (48, 64),
        "eps": 1e-3,
    },
    # launchTRowExpandAdd<aclFloat16, 16, 128, 16, 1, true>
    {
        "name": "f16_16x128",
        "dtype": np.float16,
        "src0_shape": (16, 128),      # src0eqdst=true
        "src0_valid_shape": (16, 128),
        "src1_shape": (16, 16),       # physical: 16
        "src1_valid_shape": (16, 1),  # src1Col=1
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 128),
        "eps": 1e-3,
    },
    # Note: launchTRowExpandAdd2 with src1Col=8 has different semantics - TBD
    # launchTRowExpandAdd2<float, 24, 64, 24, 8, true> - needs investigation
    # launchTRowExpandAdd2<float, 20, 64, 20, 8, false> - needs investigation
    # launchTRowExpandAdd<aclFloat16, 32, 64, 32, 1, false>
    {
        "name": "f16_32x64_noeq",
        "dtype": np.float16,
        "src0_shape": (32, 64),       # src0eqdst=false, but src0 shape still matches dst
        "src0_valid_shape": (32, 64),
        "src1_shape": (32, 16),       # physical: 16
        "src1_valid_shape": (32, 1),  # src1Col=1
        "dst_shape": (32, 64),
        "dst_valid_shape": (32, 64),
        "eps": 1e-3,
    },
    # launchTRowExpandAdd<int32_t, 16, 32, 16, 1, true>
    {
        "name": "i32_16x32",
        "dtype": np.int32,
        "src0_shape": (16, 32),       # src0eqdst=true
        "src0_valid_shape": (16, 32),
        "src1_shape": (16, 8),        # physical: 32/sizeof(i32)=8
        "src1_valid_shape": (16, 1),  # src1Col=1
        "dst_shape": (16, 32),
        "dst_valid_shape": (16, 32),
        "eps": 0,
    },
    # launchTRowExpandAdd<int16_t, 16, 64, 16, 1, true>
    {
        "name": "i16_16x64",
        "dtype": np.int16,
        "src0_shape": (16, 64),       # src0eqdst=true
        "src0_valid_shape": (16, 64),
        "src1_shape": (16, 16),       # physical: 32/sizeof(i16)=16
        "src1_valid_shape": (16, 1),  # src1Col=1
        "dst_shape": (16, 64),
        "dst_valid_shape": (16, 64),
        "eps": 0,
    },
]