#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import numpy as np
from cases import CASES
from st_common import validate_cases, save_case_data

validate_cases(CASES)

np.random.seed(42)

for case in CASES:
    dtype = case["dtype"]
    row = case["shape"][0]
    valid_row = case["valid_shape"][0]
    col = case["shape"][1]
    valid_col = case["valid_shape"][1]

    if np.issubdtype(dtype, np.integer):
        if dtype == np.int32:
            input_arr = np.random.randint(low=-100, high=100, size=(row, col)).astype(dtype)
        elif dtype == np.int16:
            input_arr = np.random.randint(low=-50, high=50, size=(row, col)).astype(dtype)
        else:
            input_arr = np.random.randint(low=-10, high=10, size=(row, col)).astype(dtype)
    else:
        input_arr = np.random.uniform(low=-1, high=1, size=(row, col)).astype(dtype)

    output_arr = np.zeros((row,))
    for i in range(valid_row):
        for j in range(valid_col):
            output_arr[i] += input_arr[i, j]
    output_arr = output_arr.astype(dtype)

    save_case_data(case["name"], {"input": input_arr, "golden": output_arr})
    print(f"[INFO] gen_data: {case['name']} shape=({row},{col}) valid=({valid_row},{valid_col}) dtype={dtype.__name__}")
