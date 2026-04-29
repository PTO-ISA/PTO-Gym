#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for trowexpand ST test cases.

trowexpand: row broadcast operation.
- Input: (rows, 1) - one scalar per row
- Output: (rows, cols) - broadcast each scalar across the entire row
"""

import numpy as np
from cases import CASES
from st_common import setup_case_rng, save_case_data

# Inline validation for multi-input format (trowexpand uses src0/dst only)
REQUIRED_KEYS = {"name", "dtype", "src0_shape", "src0_valid_shape", "dst_shape", "dst_valid_shape"}
for i, case in enumerate(CASES):
    missing = REQUIRED_KEYS - case.keys()
    if missing:
        raise ValueError(f"cases[{i}] ({case.get('name', '?')}) missing keys: {missing}")

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    src0_shape = case["src0_shape"]          # Physical shape (rows, 8)
    src0_valid_shape = case["src0_valid_shape"]  # Valid shape (rows, 1)
    dst_shape = case["dst_shape"]
    dst_valid_shape = case["dst_valid_shape"]

    # Generate input: random values for each row's scalar, padded to 8 columns
    # Physical layout: (rows, 8), but only column 0 is valid data
    input_data = np.zeros(src0_shape, dtype=dtype)
    src_vr = src0_valid_shape[0]
    input_data[:src_vr, 0] = np.random.randint(1, 10, size=src_vr).astype(dtype)

    # Generate golden: broadcast each row's scalar across columns
    # dst[i, :] = src[i, 0] for all columns
    golden = np.zeros(dst_shape, dtype=dtype)
    dst_vr, dst_vc = dst_valid_shape
    golden[:dst_vr, :dst_vc] = np.broadcast_to(input_data[:src_vr, 0:1], (dst_vr, dst_vc)).astype(dtype, copy=False)

    save_case_data(case["name"], {"input": input_data, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} src0_shape={src0_shape} dst_shape={dst_shape} dtype={dtype.__name__}")