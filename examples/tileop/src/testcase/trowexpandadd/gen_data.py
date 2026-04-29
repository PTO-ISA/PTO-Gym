#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for trowexpandadd ST test cases.

trowexpandadd: dst = src0 + broadcast(src1) across columns.
- src1Col=1: only first column valid, broadcast to all dst columns
- src1Col>1: each src1 column maps to a block of dst columns (dstCol/src1Col)
"""

import numpy as np
from cases import CASES
from st_common import setup_case_rng, save_case_data

# Inline validation for multi-input format (trowexpandadd uses src0/src1/dst)
REQUIRED_KEYS = {"name", "dtype", "src0_shape", "src0_valid_shape", "src1_shape",
                 "src1_valid_shape", "dst_shape", "dst_valid_shape"}
for i, case in enumerate(CASES):
    missing = REQUIRED_KEYS - case.keys()
    if missing:
        raise ValueError(f"cases[{i}] ({case.get('name', '?')}) missing keys: {missing}")

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    src0_shape = case["src0_shape"]
    src0_valid_shape = case["src0_valid_shape"]
    src1_shape = case["src1_shape"]
    src1_valid_shape = case["src1_valid_shape"]
    dst_shape = case["dst_shape"]
    dst_valid_shape = case["dst_valid_shape"]

    # Generate inputs
    input1 = np.random.randint(1, 10, size=src0_shape).astype(dtype)  # src0 matrix
    input2 = np.random.randint(1, 10, size=src1_shape).astype(dtype)  # src1 row vectors

    # Generate golden: dst = src0 + broadcast(src1)
    golden = np.zeros(dst_shape, dtype=dtype)
    dst_vr, dst_vc = dst_valid_shape
    src0_vr, src0_vc = src0_valid_shape
    src1_vr, src1_vc = src1_valid_shape

    if src1_vc == 1:
        # src1Col=1: broadcast first column to all dst columns
        golden[:dst_vr, :dst_vc] = (
            input1[:src0_vr, :src0_vc] + input2[:src1_vr, 0:1]
        ).astype(dtype, copy=False)
    else:
        # src1Col>1: each src1 column maps to dstCol/src1_vc columns
        # dst[:, block*repeat:(block+1)*repeat] = src0 + src1[:, block:block+1]
        repeat = dst_vc // src1_vc
        for block in range(src1_vc):
            start_col = block * repeat
            end_col = min((block + 1) * repeat, dst_vc)
            golden[:dst_vr, start_col:end_col] = (
                input1[:src0_vr, start_col:end_col] + input2[:src1_vr, block:block+1]
            ).astype(dtype, copy=False)

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} src0={src0_shape} src1={src1_shape} dst={dst_shape} dtype={dtype.__name__}")