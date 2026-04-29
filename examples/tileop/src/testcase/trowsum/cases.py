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
        "name": "case1",
        "dtype": np.float32,
        "shape": (127, 64),
        "valid_shape": (127, 63),
        "eps": 1e-3,
    },
    {
        "name": "case2",
        "dtype": np.float32,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 1e-3,
    },
    {
        "name": "case3",
        "dtype": np.float32,
        "shape": (31, 128),
        "valid_shape": (31, 127),
        "eps": 1e-3,
    },
    {
        "name": "case4",
        "dtype": np.float32,
        "shape": (15, 192),
        "valid_shape": (15, 192),
        "eps": 1e-3,
    },
    {
        "name": "case5",
        "dtype": np.float32,
        "shape": (7, 448),
        "valid_shape": (7, 447),
        "eps": 1e-3,
    },
    {
        "name": "case6",
        "dtype": np.float16,
        "shape": (256, 16),
        "valid_shape": (256, 15),
        "eps": 5e-3,
    },
    {
        "name": "case7",
        "dtype": np.float32,
        "shape": (64, 128),
        "valid_shape": (64, 128),
        "eps": 1e-3,
    },
    {
        "name": "case8",
        "dtype": np.float32,
        "shape": (32, 256),
        "valid_shape": (32, 256),
        "eps": 1e-3,
    },
    {
        "name": "case9",
        "dtype": np.float32,
        "shape": (16, 512),
        "valid_shape": (16, 512),
        "eps": 1e-3,
    },
    {
        "name": "case10",
        "dtype": np.float32,
        "shape": (8, 1024),
        "valid_shape": (8, 1024),
        "eps": 1e-3,
    },

    # int32 cases (case11-case15 from pto-isa)
    {
        "name": "case11",
        "dtype": np.int32,
        "shape": (127, 64),
        "valid_shape": (127, 63),
        "eps": 0,
    },
    {
        "name": "case12",
        "dtype": np.int32,
        "shape": (63, 64),
        "valid_shape": (63, 64),
        "eps": 0,
    },
    {
        "name": "case13",
        "dtype": np.int32,
        "shape": (31, 128),
        "valid_shape": (31, 127),
        "eps": 0,
    },
    {
        "name": "case14",
        "dtype": np.int32,
        "shape": (15, 192),
        "valid_shape": (15, 192),
        "eps": 0,
    },
    {
        "name": "case15",
        "dtype": np.int32,
        "shape": (7, 448),
        "valid_shape": (7, 447),
        "eps": 0,
    },

    # int16 cases (case16-case20 from pto-isa)
    {
        "name": "case16",
        "dtype": np.int16,
        "shape": (128, 64),
        "valid_shape": (128, 64),
        "eps": 0,
    },
    {
        "name": "case17",
        "dtype": np.int16,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 0,
    },
    {
        "name": "case18",
        "dtype": np.int16,
        "shape": (32, 128),
        "valid_shape": (32, 128),
        "eps": 0,
    },
    {
        "name": "case19",
        "dtype": np.int16,
        "shape": (16, 192),
        "valid_shape": (16, 192),
        "eps": 0,
    },
    {
        "name": "case20",
        "dtype": np.int16,
        "shape": (8, 448),
        "valid_shape": (8, 448),
        "eps": 0,
    },
]
