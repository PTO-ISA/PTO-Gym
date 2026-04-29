# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for texpands ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - shape:       (rows, cols) — allocated tile dimensions.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
  - scalar:      the scalar value to broadcast to the tile.
  - eps:         tolerance for numpy.allclose (atol and rtol).

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    # ========== float32 cases ==========
    # Full valid shape cases
    {
        "name": "f32_16x64_scalar5",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "scalar": 5.0,
        "eps": 1e-6,
    },
    {
        "name": "f32_32x32_scalar3",
        "dtype": np.float32,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "scalar": 3.0,
        "eps": 1e-6,
    },
    {
        "name": "f32_64x64_scalar2",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "scalar": 2.0,
        "eps": 1e-6,
    },
    # Partial valid shape cases
    {
        "name": "f32_16x64_partial",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (12, 48),
        "scalar": 7.0,
        "eps": 1e-6,
    },
    {
        "name": "f32_64x64_valid_60x60",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (60, 60),
        "scalar": 42.0,
        "eps": 1e-6,
    },

    # ========== int32 cases ==========
    {
        "name": "i32_64x64_scalar100",
        "dtype": np.int32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "scalar": 100,
        "eps": 0,  # exact match for integers
    },
    {
        "name": "i32_64x64_valid_60x60",
        "dtype": np.int32,
        "shape": (64, 64),
        "valid_shape": (60, 60),
        "scalar": 99,
        "eps": 0,
    },

    # ========== half (fp16) cases ==========
    {
        "name": "f16_64x64_scalar1_5",
        "dtype": np.float16,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "scalar": 1.5,
        "eps": 1e-3,  # fp16 has lower precision
    },
    {
        "name": "f16_2x4096_valid_1x3600",
        "dtype": np.float16,
        "shape": (2, 4096),
        "valid_shape": (1, 3600),
        "scalar": 2.5,
        "eps": 1e-3,
    },

    # ========== int16 cases ==========
    {
        "name": "i16_64x64_scalar50",
        "dtype": np.int16,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "scalar": 50,
        "eps": 0,
    },
    {
        "name": "i16_20x512_valid_16x200",
        "dtype": np.int16,
        "shape": (20, 512),
        "valid_shape": (16, 200),
        "scalar": 25,
        "eps": 0,
    },
]