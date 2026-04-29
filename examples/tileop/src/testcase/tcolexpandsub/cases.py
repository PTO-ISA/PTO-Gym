#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolexpandsub ST test cases.
Matches PTO-ISA testcase definitions in /home/zhoushaofan/code/pto-isa/tests/npu/a5/src/st/testcase/tcolexpandsub/

TCOLEXPANDSUB: subtract src0 by expanded src1 (broadcast src1 first row).
  - src0_shape: (src0_row, cols)   - first input tile
  - src1_shape: (src1_row, cols)   - second input tile (only first row used for broadcast)
  - dst_shape: (dst_row, cols)     - result output
  - shape: (dst_row, cols)         - alias of dst_shape, for compare.py compatibility
  - valid_shape: (valid_row, valid_col) - effective computation region

Golden: src0 - np.tile(src1, (reps, 1))[:, :dst_col]  # expand then subtract

Case naming: {dtype}_{src0_row}_{cols}_{src1_row}_{dst_col}
"""

import numpy as np

CASES = [
    {
        "name": "fp32_6_128_1_128",
        "dtype": np.float32,
        "src0_shape": (6, 128),
        "src1_shape": (1, 128),
        "shape": (6, 128),
        "valid_shape": (6, 128),
        "eps": 1e-6,
    },
    {
        "name": "fp32_18_32_1_32",
        "dtype": np.float32,
        "src0_shape": (18, 32),
        "src1_shape": (1, 32),
        "shape": (18, 32),
        "valid_shape": (18, 32),
        "eps": 1e-6,
    },
    {
        "name": "fp16_10_256_1_256",
        "dtype": np.float16,
        "src0_shape": (10, 256),
        "src1_shape": (1, 256),
        "shape": (10, 256),
        "valid_shape": (10, 256),
        "eps": 1e-3,
    },
    {
        "name": "fp16_12_64_1_64",
        "dtype": np.float16,
        "src0_shape": (12, 64),
        "src1_shape": (1, 64),
        "shape": (12, 64),
        "valid_shape": (12, 64),
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