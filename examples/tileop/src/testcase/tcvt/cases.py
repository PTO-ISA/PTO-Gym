#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcvt ST test cases.

Current TileLib tcvt support covered by this testcase:
  - f32 -> i32
  - f32 -> f16
  - f16 -> f32
  - i32 -> f32

`dtype` is kept for shared validation compatibility.
Actual data generation and comparison use `src_dtype` / `dst_dtype`.
"""

import numpy as np

CASES = [
    {
        "name": "f32_to_i32_rint_16x64",
        "dtype": np.int32,
        "src_dtype": np.float32,
        "dst_dtype": np.int32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 0.0,
    },
    {
        "name": "f32_to_i32_round_16x64",
        "dtype": np.int32,
        "src_dtype": np.float32,
        "dst_dtype": np.int32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "ROUND",
        "eps": 0.0,
    },
    {
        "name": "i32_to_f32_rint_16x64",
        "dtype": np.float32,
        "src_dtype": np.int32,
        "dst_dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-6,
    },
    {
        "name": "f32_to_f16_rint_16x64",
        "dtype": np.float16,
        "src_dtype": np.float32,
        "dst_dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-3,
    },
    {
        "name": "f16_to_f32_rint_16x64",
        "dtype": np.float32,
        "src_dtype": np.float16,
        "dst_dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-6,
    },
]
