#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trowsum ST test cases.

Aligned with pto-isa tests/npu/a5/src/st/testcase/trowsum (20 cases).
"""

import numpy as np

CASES = [
    # f32 cases (case1-case10 from pto-isa)
    {
        "name": "f32_127x64_valid127x63",
        "dtype": np.float32,
        "shape": (127, 64),
        "valid_shape": (127, 63),
        "eps": 1e-3,
    },
    {
        "name": "f32_63x64",
        "dtype": np.float32,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 1e-3,
    },
    {
        "name": "f32_31x128_valid31x127",
        "dtype": np.float32,
        "shape": (31, 128),
        "valid_shape": (31, 127),
        "eps": 1e-3,
    },
    {
        "name": "f32_15x192",
        "dtype": np.float32,
        "shape": (15, 192),
        "valid_shape": (15, 192),
        "eps": 1e-3,
    },
    {
        "name": "f32_7x448_valid7x447",
        "dtype": np.float32,
        "shape": (7, 448),
        "valid_shape": (7, 447),
        "eps": 1e-3,
    },
    {
        "name": "f16_256x16_valid256x15",
        "dtype": np.float16,
        "shape": (256, 16),
        "valid_shape": (256, 15),
        "eps": 5e-3,
    },
    {
        "name": "f32_64x128",
        "dtype": np.float32,
        "shape": (64, 128),
        "valid_shape": (64, 128),
        "eps": 1e-3,
    },
    {
        "name": "f32_32x256",
        "dtype": np.float32,
        "shape": (32, 256),
        "valid_shape": (32, 256),
        "eps": 1e-3,
    },
    {
        "name": "f32_16x512",
        "dtype": np.float32,
        "shape": (16, 512),
        "valid_shape": (16, 512),
        "eps": 1e-3,
    },
    {
        "name": "f32_8x1024",
        "dtype": np.float32,
        "shape": (8, 1024),
        "valid_shape": (8, 1024),
        "eps": 1e-3,
    },

    # int32 cases (case11-case15 from pto-isa)
    {
        "name": "i32_127x64_valid127x63",
        "dtype": np.int32,
        "shape": (127, 64),
        "valid_shape": (127, 63),
        "eps": 0,
    },
    {
        "name": "i32_63x64",
        "dtype": np.int32,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 0,
    },
    {
        "name": "i32_31x128_valid31x127",
        "dtype": np.int32,
        "shape": (31, 128),
        "valid_shape": (31, 127),
        "eps": 0,
    },
    {
        "name": "i32_15x192",
        "dtype": np.int32,
        "shape": (15, 192),
        "valid_shape": (15, 192),
        "eps": 0,
    },
    {
        "name": "i32_7x448_valid7x447",
        "dtype": np.int32,
        "shape": (7, 448),
        "valid_shape": (7, 447),
        "eps": 0,
    },

    # int16 cases (case16-case20 from pto-isa)
    {
        "name": "i16_128x64",
        "dtype": np.int16,
        "shape": (128, 64),
        "valid_shape": (128, 64),
        "eps": 0,
    },
    {
        "name": "i16_64x64",
        "dtype": np.int16,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 0,
    },
    {
        "name": "i16_32x128",
        "dtype": np.int16,
        "shape": (32, 128),
        "valid_shape": (32, 128),
        "eps": 0,
    },
    {
        "name": "i16_16x192",
        "dtype": np.int16,
        "shape": (16, 192),
        "valid_shape": (16, 192),
        "eps": 0,
    },
    {
        "name": "i16_8x448",
        "dtype": np.int16,
        "shape": (8, 448),
        "valid_shape": (8, 448),
        "eps": 0,
    },
    # i16 overflow case to test vcvt NOSAT behavior
    {
        "name": "i16_1x64_overflow",
        "dtype": np.int16,
        "shape": (1, 64),
        "valid_shape": (1, 64),
        "eps": 0,
        "overflow": True,
    },
]
