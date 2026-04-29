#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolexpandexpdif ST test cases.
Matches PTO-ISA testcase definitions in /home/zhoushaofan/code/pto-isa/tests/npu/a5/src/st/testcase/tcolexpandexpdif/

TCOLEXPANDEXPDIF: compute exp(src0) - exp(expanded_src1) where src1 is expanded by tiling.
  - src0_shape: (src0_row, cols)    - first input tile
  - src1_shape: (src1_row, cols)    - second input tile (tiled to match src0 rows)
  - dst_shape: (dst_row, dst_col)   - output tile
  - shape: (dst_row, dst_col)       - alias of dst_shape, for compare.py compatibility
  - valid_shape: (valid_row, valid_col) - effective computation region

Golden: np.exp(src0) - np.exp(np.tile(src1, (reps, 1))[:, :dst_col])
  where reps = dst_row // src1_row

Case naming: {dtype}_{src0_row}_{src0_col}_{src1_row}_{src1_col}
"""

import numpy as np

CASES = [
    {
        "name": "fp32_32_16_1_16",
        "dtype": np.float32,
        "src0_shape": (32, 16),
                "src1_shape": (1, 16),
        "shape": (32, 16),
        "valid_shape": (32, 16),
        "eps": 1e-5,
    },
    {
        "name": "fp32_16_32_1_32",
        "dtype": np.float32,
        "src0_shape": (16, 32),
                "src1_shape": (1, 32),
        "shape": (16, 32),
        "valid_shape": (16, 32),
        "eps": 1e-5,
    },
    {
        "name": "fp16_32_32_1_32",
        "dtype": np.float16,
        "src0_shape": (32, 32),
                "src1_shape": (1, 32),
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-2,
    },
    {
        "name": "fp16_16_128_1_128",
        "dtype": np.float16,
        "src0_shape": (16, 128),
                "src1_shape": (1, 128),
        "shape": (16, 128),
        "valid_shape": (16, 128),
        "eps": 1e-2,
    },
]