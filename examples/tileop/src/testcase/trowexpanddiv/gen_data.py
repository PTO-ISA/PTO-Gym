#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for trowexpanddiv ST test cases.

trowexpanddiv: dst = src0 / broadcast(src1)
"""

import numpy as np
from cases import CASES
from st_common import setup_case_rng, save_case_data

# Inline validation for multi-input format (trowexpanddiv uses src0/src1/dst)
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

    input1 = np.random.randint(1, 10, size=src0_shape).astype(dtype)
    input2 = np.random.randint(1, 10, size=src1_shape).astype(dtype)

    golden = np.zeros(dst_shape, dtype=dtype)
    dst_vr, dst_vc = dst_valid_shape
    src0_vr, src0_vc = src0_valid_shape
    src1_vr, src1_vc = src1_valid_shape

    # Compute golden based on src1Col semantics
    # src1Col=1: broadcast single column to all dst columns
    # src1Col>1: each src1 column broadcasts to dst_vc/src1_vc columns
    if dtype in (np.int8, np.int16, np.int32):
        if src1_vc == 1:
            golden[:dst_vr, :dst_vc] = (
                input1[:src0_vr, :src0_vc] // input2[:src1_vr, 0:1]
            ).astype(dtype, copy=False)
        else:
            # src1Col > 1: each src1 column broadcasts to dst_vc/src1_vc dst columns
            block_size = dst_vc // src1_vc
            for c in range(src1_vc):
                golden[:dst_vr, c*block_size:(c+1)*block_size] = (
                    input1[:src0_vr, c*block_size:(c+1)*block_size] // input2[:src1_vr, c:c+1]
                ).astype(dtype, copy=False)
    else:
        if src1_vc == 1:
            golden[:dst_vr, :dst_vc] = (
                input1[:src0_vr, :src0_vc] / input2[:src1_vr, 0:1]
            ).astype(dtype, copy=False)
        else:
            # src1Col > 1: each src1 column broadcasts to dst_vc/src1_vc dst columns
            block_size = dst_vc // src1_vc
            for c in range(src1_vc):
                golden[:dst_vr, c*block_size:(c+1)*block_size] = (
                    input1[:src0_vr, c*block_size:(c+1)*block_size] / input2[:src1_vr, c:c+1]
                ).astype(dtype, copy=False)

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} dtype={dtype.__name__}")