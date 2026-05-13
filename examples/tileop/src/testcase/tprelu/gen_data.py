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
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]

    rows, cols = shape
    vr, vc = valid_shape

    input0 = np.random.uniform(-8, high=8, size=(rows, cols)).astype(dtype)
    input1 = np.random.uniform(-8, high=8, size=(rows, cols)).astype(dtype)

    golden = np.zeros((rows, cols), dtype=dtype)
    for i in range(vr):
        for j in range(vc):
            if input0[i, j] > 0:
                golden[i, j] = input0[i, j]
            else:
                golden[i, j] = dtype(input0[i, j] * input1[i, j])

    save_case_data(case["name"], {"input0": input0, "input1": input1, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__}")