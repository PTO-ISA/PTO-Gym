#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
import struct
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    dst_shape = case["dst_shape"]
    valid_shape = case["valid_shape"]

    rows, cols = shape
    dst_rows, dst_cols = dst_shape
    vr, vc = valid_shape

    input_arr = np.random.uniform(low=-8, high=8, size=(rows, cols)).astype(dtype)
    slope = np.random.uniform(low=-8, high=8, size=(1, 1)).astype(np.float32)
    golden = np.zeros((dst_rows, dst_cols), dtype=dtype)

    for i in range(vr):
        for j in range(vc):
            if input_arr[i, j] > 0:
                golden[i, j] = input_arr[i, j]
            else:
                golden[i, j] = dtype(input_arr[i, j] * slope[0, 0])

    slope_arr = np.array([slope[0, 0]], dtype=np.float32)

    save_case_data(case["name"], {"input": input_arr, "slope": slope_arr, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} dst_shape={dst_shape} valid_shape={valid_shape} dtype={dtype.__name__}")