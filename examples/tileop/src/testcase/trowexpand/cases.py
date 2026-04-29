#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trowexpand ST test cases.

trowexpand is a row broadcast operation: expands a scalar per row to the entire row.
- Input shape: (rows, srcCols) - physical layout for NPU alignment
- srcCols = 32/sizeof(dtype) for 32-byte alignment
- Output shape: (rows, dstCols) - broadcast each scalar across the row
- dstValidCols may be less than dstCols for partial valid region

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32, np.float16, np.int8).
  - src0_shape:   (rows, srcCols) — physical input tile dimensions.
  - src0_valid_shape: (valid_rows, 1) — effective input region.
  - dst_shape:   (rows, dstCols) — output tile dimensions.
  - dst_valid_shape: (valid_rows, valid_cols) — effective output region.
  - eps:         tolerance for numpy.allclose (atol and rtol).
"""

import numpy as np

CASES = [
    # f32 cases (srcCols=8 for 32-byte alignment)
    {
        "name": "f32_16x128",
        "dtype": np.float32,
        "src0_shape": (16, 8),
        "src0_valid_shape": (16, 1),
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 128),
        "eps": 1e-6,
    },
    {
        "name": "f32_16x127",
        "dtype": np.float32,
        "src0_shape": (16, 8),
        "src0_valid_shape": (16, 1),
        "dst_shape": (16, 128),
        "dst_valid_shape": (16, 127),  # partial valid region
        "eps": 1e-6,
    },
    # f16 cases (srcCols=16 for 32-byte alignment)
    {
        "name": "f16_16x512",
        "dtype": np.float16,
        "src0_shape": (16, 16),
        "src0_valid_shape": (16, 1),
        "dst_shape": (16, 512),
        "dst_valid_shape": (16, 512),
        "eps": 1e-3,
    },
    {
        "name": "f16_16x511",
        "dtype": np.float16,
        "src0_shape": (16, 16),
        "src0_valid_shape": (16, 1),
        "dst_shape": (16, 512),
        "dst_valid_shape": (16, 511),  # partial valid region
        "eps": 1e-3,
    },
    # i8 cases (srcCols=32 for 32-byte alignment)
    {
        "name": "i8_16x256",
        "dtype": np.int8,
        "src0_shape": (16, 32),
        "src0_valid_shape": (16, 1),
        "dst_shape": (16, 256),
        "dst_valid_shape": (16, 256),
        "eps": 0,  # exact match for integers
    },
    {
        "name": "i8_16x255",
        "dtype": np.int8,
        "src0_shape": (16, 32),
        "src0_valid_shape": (16, 1),
        "dst_shape": (16, 256),
        "dst_valid_shape": (16, 255),  # partial valid region
        "eps": 0,
    },
]