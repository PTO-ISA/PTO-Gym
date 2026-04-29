#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tlrelu ST test cases.

Each case defines:
  - name:            case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:           numpy dtype (e.g. np.float32).
  - shape:           (rows, cols) — src tile dimensions (UB allocation).
  - valid_shape:     (valid_rows, valid_cols) — effective computation region.
  - dst_shape:       (rows, cols) — dst tile physical dimensions (UB allocation, may have padding).
  - dst_valid_shape: (valid_rows, valid_cols) — dst effective region (same as valid_shape).
  - eps:             tolerance for numpy.allclose (atol and rtol).

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "f32_32x64_dst128",
        "dtype": np.float32,
        "shape": (32, 64),
        "valid_shape": (32, 64),
        "dst_shape": (32, 128),
        "dst_valid_shape": (32, 64),
        "eps": 1e-3,
    },
    {
        "name": "f16_63x64_dst128",
        "dtype": np.float16,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "dst_shape": (63, 128),
        "dst_valid_shape": (63, 64),
        "eps": 1e-3,
    },
    {
        "name": "f32_7x448_dst512",
        "dtype": np.float32,
        "shape": (7, 448),
        "valid_shape": (7, 448),
        "dst_shape": (7, 512),
        "dst_valid_shape": (7, 448),
        "eps": 1e-3,
    },
    {
        "name": "f32_256x16_dst32",
        "dtype": np.float32,
        "shape": (256, 16),
        "valid_shape": (256, 16),
        "dst_shape": (256, 32),
        "dst_valid_shape": (256, 16),
        "eps": 1e-3,
    },
]