#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trowexpanddiv ST test cases.

trowexpanddiv: dst = src0 / broadcast(src1) across columns.
- src1Col determines how src1 is broadcast:
  - src1Col=1: only first column is valid, broadcast to dstCols
  - src1Col>1: each src1 column maps to a block of dst columns (dstCol/src1Col columns per src1 value)
- src1 physical cols = 32/sizeof(dtype) for NPU alignment
- highPrecision: use high precision mode for computation
"""

import numpy as np

CASES = [
    # launchTRowExpandDiv<float, 40, 64, 40, 1, true, false>
    {
        "name": "f32_40x64",
        "dtype": np.float32,
        "src0_shape": (40, 64),       # src0eqdst=true
        "src0_valid_shape": (40, 64),
        "src1_shape": (40, 8),        # physical: 32/sizeof(f32)=8
        "src1_valid_shape": (40, 1),  # src1Col=1
        "dst_shape": (40, 64),
        "dst_valid_shape": (40, 64),
        "eps": 1e-6,
        "high_precision": False,
    },
    # launchTRowExpandDiv<float, 16, 256, 16, 1, true, false>
    {
        "name": "f32_16x256",
        "dtype": np.float32,
        "src0_shape": (16, 256),
        "src0_valid_shape": (16, 256),
        "src1_shape": (16, 8),
        "src1_valid_shape": (16, 1),
        "dst_shape": (16, 256),
        "dst_valid_shape": (16, 256),
        "eps": 1e-6,
        "high_precision": False,
    },
    # launchTRowExpandDiv<aclFloat16, 16, 32, 16, 1, true, false>
    {
        "name": "f16_16x32",
        "dtype": np.float16,
        "src0_shape": (16, 32),
        "src0_valid_shape": (16, 32),
        "src1_shape": (16, 16),       # physical: 32/sizeof(f16)=16
        "src1_valid_shape": (16, 1),
        "dst_shape": (16, 32),
        "dst_valid_shape": (16, 32),
        "eps": 1e-3,
        "high_precision": False,
    },
    # launchTRowExpandDiv<aclFloat16, 32, 512, 32, 1, true, false>
    {
        "name": "f16_32x512",
        "dtype": np.float16,
        "src0_shape": (32, 512),
        "src0_valid_shape": (32, 512),
        "src1_shape": (32, 16),
        "src1_valid_shape": (32, 1),
        "dst_shape": (32, 512),
        "dst_valid_shape": (32, 512),
        "eps": 1e-3,
        "high_precision": False,
    },
    # launchTRowExpandDiv<float, 16, 128, 16, 1, false, false>
    {
        "name": "f32_16x128_noeq",
        "dtype": np.float32,
        "src0_shape": (16, 128),      # src0eqdst=false
        "src0_valid_shape": (16, 128),
        "src1_shape": (16, 8),
        "src1_valid_shape": (16, 1),
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 128),
        "eps": 1e-6,
        "high_precision": False,
    },
    # launchTRowExpandDiv<aclFloat16, 32, 64, 32, 1, false, false>
    {
        "name": "f16_32x64_noeq",
        "dtype": np.float16,
        "src0_shape": (32, 64),
        "src0_valid_shape": (32, 64),
        "src1_shape": (32, 16),
        "src1_valid_shape": (32, 1),
        "dst_shape": (32, 64),
        "dst_valid_shape": (32, 64),
        "eps": 1e-3,
        "high_precision": False,
    },
    # launchTRowExpandDiv<float, 40, 32, 40, 1, true, true>
    {
        "name": "f32_40x32_hp",
        "dtype": np.float32,
        "src0_shape": (40, 32),
        "src0_valid_shape": (40, 32),
        "src1_shape": (40, 8),
        "src1_valid_shape": (40, 1),
        "dst_shape": (40, 32),
        "dst_valid_shape": (40, 32),
        "eps": 1e-6,
        "high_precision": True,
    },
    # launchTRowExpandDiv<aclFloat16, 16, 128, 16, 1, true, true>
    {
        "name": "f16_16x128_hp",
        "dtype": np.float16,
        "src0_shape": (16, 128),
        "src0_valid_shape": (16, 128),
        "src1_shape": (16, 16),
        "src1_valid_shape": (16, 1),
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 128),
        "eps": 1e-3,
        "high_precision": True,
    },
    # Note: launchTRowExpandDiv2 with src1Col>1 has different semantics - TBD
]