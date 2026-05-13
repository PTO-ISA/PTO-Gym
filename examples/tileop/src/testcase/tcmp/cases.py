# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcmp ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - shape:       (rows, cols) — allocated tile dimensions (same for src and dst).
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
  - dst_dtype:   output mask dtype (i8 - packed mask, same shape as input).
  - cmp_mode:    comparison mode: "eq", "ne", "lt", "gt", "ge", "le".
  - eps:         tolerance (exact match for masks, eps=0).

Aligned with testcase/tcmp test cases.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    # Case 1: f16 32x32 EQ (half_32x32_32x32)
    {
        "name": "f16_32x32_eq",
        "dtype": np.float16,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "dst_dtype": np.int8,
        "cmp_mode": "eq",
        "eps": 0,
    },
    # Case 2: f32 8x64 GT (float_8x64_8x64)
    {
        "name": "f32_8x64_gt",
        "dtype": np.float32,
        "shape": (8, 64),
        "valid_shape": (8, 64),
        "dst_dtype": np.int8,
        "cmp_mode": "gt",
        "eps": 0,
    },
    # Case 3: i32 4x64 NE (int32_4x64_4x64)
    {
        "name": "i32_4x64_ne",
        "dtype": np.int32,
        "shape": (4, 64),
        "valid_shape": (4, 64),
        "dst_dtype": np.int8,
        "cmp_mode": "ne",
        "eps": 0,
    },
    # Case 4: i32 128x128 LT with valid 64x64 (int32_128x128_64x64)
    {
        "name": "i32_128x128_lt",
        "dtype": np.int32,
        "shape": (128, 128),
        "valid_shape": (64, 64),
        "dst_dtype": np.int8,
        "cmp_mode": "lt",
        "eps": 0,
    },
    # Case 5: i32 64x64 EQ with valid 32x32 (int32_64x64_32x32)
    {
        "name": "i32_64x64_eq",
        "dtype": np.int32,
        "shape": (64, 64),
        "valid_shape": (32, 32),
        "dst_dtype": np.int8,
        "cmp_mode": "eq",
        "eps": 0,
    },
    # Case 6: i32 16x32 EQ (int32_16x32_16x32)
    {
        "name": "i32_16x32_eq",
        "dtype": np.int32,
        "shape": (16, 32),
        "valid_shape": (16, 32),
        "dst_dtype": np.int8,
        "cmp_mode": "eq",
        "eps": 0,
    },
    # Case 7: f32 128x128 LE with valid 64x64 (float_128x128_64x64)
    {
        "name": "f32_128x128_le",
        "dtype": np.float32,
        "shape": (128, 128),
        "valid_shape": (64, 64),
        "dst_dtype": np.int8,
        "cmp_mode": "le",
        "eps": 0,
    },
    # Case 8: i32 77x80 EQ with valid 32x32 (int32_77x80_32x32)
    {
        "name": "i32_77x80_eq",
        "dtype": np.int32,
        "shape": (77, 80),
        "valid_shape": (32, 32),
        "dst_dtype": np.int8,
        "cmp_mode": "eq",
        "eps": 0,
    },
    # Case 9: i32 32x32 EQ (int32_32x32_32x32)
    {
        "name": "i32_32x32_eq",
        "dtype": np.int32,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "dst_dtype": np.int8,
        "cmp_mode": "eq",
        "eps": 0,
    },
    # Case 10: i16 32x32 EQ with valid 16x32 (int16_32x32_16x32)
    {
        "name": "i16_32x32_eq",
        "dtype": np.int16,
        "shape": (32, 32),
        "valid_shape": (16, 32),
        "dst_dtype": np.int8,
        "cmp_mode": "eq",
        "eps": 0,
    },
    # Case 11: i16 77x80 LE with valid 32x32 (int16_77x80_32x32)
    {
        "name": "i16_77x80_le",
        "dtype": np.int16,
        "shape": (77, 80),
        "valid_shape": (32, 32),
        "dst_dtype": np.int8,
        "cmp_mode": "le",
        "eps": 0,
    },
]