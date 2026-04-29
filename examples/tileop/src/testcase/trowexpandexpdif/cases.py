#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trowexpandexpdif ST test cases.

trowexpandexpdif: dst = exp(src0 - broadcast(src1))
- src1Col=1: only first column of src1 is valid, broadcast to dstCols
- src1Col>1: launchTRowExpandExpdif2 with different semantics (TBD)
- src1 physical cols = 32/sizeof(dtype) for NPU alignment
- src0eqdst: true means src0 shape equals dst shape
"""

import numpy as np

CASES = [
    # launchTRowExpandExpdif<float, 32, 64, 32, 1, true>
    {
        "name": "f32_32x64",
        "dtype": np.float32,
        "src0_shape": (32, 64),
        "src0_valid_shape": (32, 64),
        "src1_shape": (32, 8),        # physical: 32/sizeof(f32)=8
        "src1_valid_shape": (32, 1),  # src1Col=1
        "dst_shape": (32, 64),
        "dst_valid_shape": (32, 64),
        "eps": 1e-5,
    },
    # launchTRowExpandExpdif<float, 16, 32, 16, 1, true>
    {
        "name": "f32_16x32",
        "dtype": np.float32,
        "src0_shape": (16, 32),
        "src0_valid_shape": (16, 32),
        "src1_shape": (16, 8),
        "src1_valid_shape": (16, 1),
        "dst_shape": (16, 32),
        "dst_valid_shape": (16, 32),
        "eps": 1e-5,
    },
    # launchTRowExpandExpdif<aclFloat16, 16, 32, 16, 1, true>
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
    },
    # launchTRowExpandExpdif<aclFloat16, 48, 64, 48, 1, true>
    {
        "name": "f16_48x64",
        "dtype": np.float16,
        "src0_shape": (48, 64),
        "src0_valid_shape": (48, 64),
        "src1_shape": (48, 16),
        "src1_valid_shape": (48, 1),
        "dst_shape": (48, 64),
        "dst_valid_shape": (48, 64),
        "eps": 1e-3,
    },
    # launchTRowExpandExpdif<float, 16, 128, 16, 1, false>
    {
        "name": "f32_16x128_noeq",
        "dtype": np.float32,
        "src0_shape": (16, 128),      # src0eqdst=false
        "src0_valid_shape": (16, 128),
        "src1_shape": (16, 8),
        "src1_valid_shape": (16, 1),
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 128),
        "eps": 1e-5,
    },
    # Note: launchTRowExpandExpdif2 with src1Col>1 has different semantics - TBD
    # - float, 24, 64, 24, 8, true (src1Col=8)
    # - aclFloat16, 16, 64, 16, 16, false (src1Col=16)
]