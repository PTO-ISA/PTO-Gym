#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolexpandadd ST test cases.

TCOLEXPANDADD: expand src1 then add with src0.
  - src0_shape: (dst_row, dst_col) - already expanded (src0_shape = shape)
  - src1_shape: (src1_row, src1_col) - to be expanded (usually src1_row=1)
  - shape: (dst_row, dst_col) - output shape
"""

import numpy as np

CASES = [
    {
        "name": "fp32_16_128_1_128",
        "dtype": np.float32,
        "src0_shape": (16, 128),
        "src1_shape": (1, 128),
        "shape": (16, 128),
        "valid_shape": (16, 128),
        "eps": 1e-3,
    },
    {
        "name": "fp32_32_32_1_32",
        "dtype": np.float32,
        "src0_shape": (32, 32),
        "src1_shape": (1, 32),
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-3,
    },
    {
        "name": "fp16_4_256_1_256",
        "dtype": np.float16,
        "src0_shape": (4, 256),
        "src1_shape": (1, 256),
        "shape": (4, 256),
        "valid_shape": (4, 256),
        "eps": 1e-3,
    },
    {
        "name": "fp16_10_64_1_64",
        "dtype": np.float16,
        "src0_shape": (10, 64),
        "src1_shape": (1, 64),
        "shape": (10, 64),
        "valid_shape": (10, 64),
        "eps": 1e-3,
    },
    {
        "name": "int32_16_32_1_32",
        "dtype": np.int32,
        "src0_shape": (16, 32),
        "src1_shape": (1, 32),
        "shape": (16, 32),
        "valid_shape": (16, 32),
        "eps": 0,
    },
    {
        "name": "int16_16_64_1_64",
        "dtype": np.int16,
        "src0_shape": (16, 64),
        "src1_shape": (1, 64),
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 0,
    },
]