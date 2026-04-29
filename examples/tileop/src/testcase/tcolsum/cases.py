#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolsum ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - shape:       (rows, cols) — allocated tile dimensions for input.
  - valid_shape: (valid_rows, valid_cols) — effective computation region for input.
  - dst_shape:       (1, cols) — allocated tile dimensions for output.
  - dst_valid_shape: (1, valid_cols) — effective computation region for output.
  - eps:         tolerance for numpy.allclose (atol and rtol).

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "f32_1x256",
        "dtype": np.float32,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "eps": 1e-6,
    },
    {
        "name": "f32_16x128",
        "dtype": np.float32,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "eps": 1e-6,
    },
    {
        "name": "f32_16x256",
        "dtype": np.float32,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "eps": 1e-6,
    },
    {
        "name": "f32_64x128_1",
        "dtype": np.float32,
        "shape": (64, 128),
        "valid_shape": (63, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "eps": 1e-6,
    },
    {
        "name": "f32_64x128_2",
        "dtype": np.float32,
        "shape": (64, 128),
        "valid_shape": (64, 128),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 128),
        "eps": 1e-6,
    },
    {
        "name": "f32_1x512",
        "dtype": np.float32,
        "shape": (1, 512),
        "valid_shape": (1, 511),
        "dst_shape": (1, 512),
        "dst_valid_shape": (1, 511),
        "eps": 1e-6,
    },
    {
        "name": "f16_1x256",
        "dtype": np.float16,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "eps": 1e-3,
    },
    {
        "name": "f16_16x128",
        "dtype": np.float16,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "eps": 1e-3,
    },
    {
        "name": "f16_16x256",
        "dtype": np.float16,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "eps": 1e-3,
    },
    {
        "name": "f16_64x128_1",
        "dtype": np.float16,
        "shape": (64, 128),
        "valid_shape": (63, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "eps": 1e-3,
    },
    {
        "name": "f16_64x128_2",
        "dtype": np.float16,
        "shape": (64, 128),
        "valid_shape": (64, 128),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 128),
        "eps": 1e-3,
    },
    {
        "name": "i8_1x256",
        "dtype": np.int8,
        "shape": (1, 256),
        "valid_shape": (1, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "eps": 0,
    },
    {
        "name": "i8_16x128",
        "dtype": np.int8,
        "shape": (16, 128),
        "valid_shape": (16, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "eps": 0,
    },
    {
        "name": "i8_16x256",
        "dtype": np.int8,
        "shape": (16, 256),
        "valid_shape": (15, 255),
        "dst_shape": (1, 256),
        "dst_valid_shape": (1, 255),
        "eps": 0,
    },
    {
        "name": "i8_64x128_1",
        "dtype": np.int8,
        "shape": (64, 128),
        "valid_shape": (63, 127),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 127),
        "eps": 0,
    },
    {
        "name": "i8_64x128_2",
        "dtype": np.int8,
        "shape": (64, 128),
        "valid_shape": (64, 128),
        "dst_shape": (1, 128),
        "dst_valid_shape": (1, 128),
        "eps": 0,
    },
]
