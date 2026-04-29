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
    src0_shape = case["src0_shape"]
    dst_shape = case["shape"]
    src1_shape = case["src1_shape"]
    valid_shape = case["valid_shape"]

    src0 = np.random.uniform(-255, 255, size=src0_shape).astype(dtype)
    src1 = np.random.uniform(1, 255, size=src1_shape).astype(dtype)

    dst_row, dst_col = dst_shape
    src1_row = src1_shape[0]
    reps = (dst_row + src1_row - 1) // src1_row

    expanded_src1 = np.tile(src1, (reps, 1))[:dst_row, :dst_col]
    golden = np.exp((src0.astype(np.float64) - expanded_src1.astype(np.float64)))
    golden = golden.astype(dtype)

    save_case_data(case["name"], {"input0": src0, "input1": src1, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} src0={src0_shape} src1={src1_shape} dst={dst_shape} dtype={dtype.__name__}")