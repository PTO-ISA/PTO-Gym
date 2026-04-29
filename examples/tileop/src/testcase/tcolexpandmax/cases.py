#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolexpandmax ST test cases.
Matches PTO-ISA testcase definitions in /home/zhoushaofan/code/pto-isa/tests/npu/a5/src/st/testcase/tcolexpandmax/

TCOLEXPANDMAX: compute elementwise maximum of src0 and tiled src1.
  - src0_shape: (src0_row, cols)       - first input tile
  - src1_shape: (1, cols)              - second input tile (single row, broadcasted)
  - dst_shape: (dst_row, cols)         - output tile
  - shape: (dst_row, cols)             - alias of dst_shape, for compare.py compatibility
  - valid_shape: (valid_row, valid_col) - effective computation region
  - reps: number of times to tile src1 (equals src0_row)

Golden: np.maximum(src0, np.tile(src1, (reps, 1))[:, :dst_col])

Case naming: {dtype}_{src0_row}_{src0_col}_{src1_row}_{dst_col}
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
        "reps": 16,
        "eps": 1e-6,
    },
    {
        "name": "fp32_32_32_1_32",
        "dtype": np.float32,
        "src0_shape": (32, 32),
                "src1_shape": (1, 32),
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "reps": 32,
        "eps": 1e-6,
    },
    {
        "name": "fp16_4_256_1_256",
        "dtype": np.float16,
        "src0_shape": (4, 256),
                "src1_shape": (1, 256),
        "shape": (4, 256),
        "valid_shape": (4, 256),
        "reps": 4,
        "eps": 1e-3,
    },
    {
        "name": "fp16_10_64_1_64",
        "dtype": np.float16,
        "src0_shape": (10, 64),
                "src1_shape": (1, 64),
        "shape": (10, 64),
        "valid_shape": (10, 64),
        "reps": 10,
        "eps": 1e-3,
    },
    {
        "name": "int32_16_32_1_32",
        "dtype": np.int32,
        "src0_shape": (16, 32),
        "src1_shape": (1, 32),
        "shape": (16, 32),
        "valid_shape": (16, 32),
        "reps": 16,
        "eps": 0,
    },
    {
        "name": "int16_16_64_1_64",
        "dtype": np.int16,
        "src0_shape": (16, 64),
        "src1_shape": (1, 64),
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "reps": 16,
        "eps": 0,
    },
]