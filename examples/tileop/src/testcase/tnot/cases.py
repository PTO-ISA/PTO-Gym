#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tnot ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.int8, np.int16, np.int32).
  - shape:       (rows, cols) — allocated tile dimensions.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
  - eps:         tolerance for numpy.allclose (atol and rtol), 0 for exact match.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "int8_64x64",   
        "dtype": np.int8,   
        "shape": (64, 64), 
        "valid_shape": (64, 64), 
        "eps": 0
    },
    {
        "name": "uint8_60x60",  
        "dtype": np.uint8,  
        "shape": (64, 64), 
        "valid_shape": (60, 60), 
        "eps": 0
    },
    {
        "name": "int16_64x64",  
        "dtype": np.int16,  
        "shape": (64, 64), 
        "valid_shape": (64, 64), 
        "eps": 0
    },
    {
        "name": "uint16_60x60", 
        "dtype": np.uint16, 
        "shape": (64, 64), 
        "valid_shape": (60, 60), 
        "eps": 0
    },
    {
        "name": "int32_64x64",  
        "dtype": np.int32,  
        "shape": (64, 64), 
        "valid_shape": (64, 64), 
        "eps": 0
    },
    {
        "name": "uint32_60x60", 
        "dtype": np.uint32, 
        "shape": (64, 64), 
        "valid_shape": (60, 60), 
        "eps": 0
    },
]