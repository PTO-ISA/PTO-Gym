#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolexpand ST test cases.
Matches PTO-ISA testcase definitions in /home/zhoushaofan/code/pto-isa/tests/npu/a5/src/st/testcase/tcolexpand/

TCOLEXPAND: expand src first row to dst all rows by broadcasting.
  - src_shape: (src_row, cols)    - input tile (only first row is used for broadcast)
  - dst_shape: (dst_row, cols)    - expanded output
  - shape: (dst_row, cols)        - alias of dst_shape, for compare.py compatibility
  - valid_shape: (valid_row, valid_col) - effective computation region

Case naming: {dtype}_{src_row}_{dst_row}_{cols}_{valid_col}
"""

import numpy as np

CASES = [
    {
        "name": "half_1_16_512_512",
        "dtype": np.float16,
        "src_shape": (1, 512),
        "shape": (16, 512),
        "valid_shape": (16, 512),
        "eps": 1e-3,
    },
    {
        "name": "int8_2_32_256_255",
        "dtype": np.int8,
        "src_shape": (2, 256),
        "shape": (32, 256),
        "valid_shape": (32, 255),
        "eps": 0,
    },
    {
        "name": "float_1_8_128_63",
        "dtype": np.float32,
        "src_shape": (1, 128),
        "shape": (8, 128),
        "valid_shape": (8, 63),
        "eps": 1e-6,
    },
    {
        "name": "half_1_33_512_512",
        "dtype": np.float16,
        "src_shape": (1, 512),
        "shape": (33, 512),
        "valid_shape": (33, 512),
        "eps": 1e-3,
    },
    {
        "name": "int8_2_17_256_44",
        "dtype": np.int8,
        "src_shape": (2, 256),
        "shape": (17, 256),
        "valid_shape": (17, 44),
        "eps": 0,
    },
    {
        "name": "float_1_54_64_63",
        "dtype": np.float32,
        "src_shape": (1, 64),
        "shape": (54, 64),
        "valid_shape": (54, 63),
        "eps": 1e-6,
    },
]