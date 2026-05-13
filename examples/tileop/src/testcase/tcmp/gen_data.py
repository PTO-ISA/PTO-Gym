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
    dst_dtype = case["dst_dtype"]
    cmp_mode = case["cmp_mode"]

    # Generate random input data
    input1 = np.random.randint(1, 10, size=shape).astype(dtype)
    input2 = np.random.randint(1, 10, size=shape).astype(dtype)

    # Compute comparison mask (boolean)
    vr, vc = valid_shape
    mask_bits = np.zeros(shape, dtype=np.bool_)
    input1_valid = input1[:vr, :vc]
    input2_valid = input2[:vr, :vc]

    if cmp_mode == "eq":
        mask_bits[:vr, :vc] = (input1_valid == input2_valid)
    elif cmp_mode == "ne":
        mask_bits[:vr, :vc] = (input1_valid != input2_valid)
    elif cmp_mode == "lt":
        mask_bits[:vr, :vc] = (input1_valid < input2_valid)
    elif cmp_mode == "gt":
        mask_bits[:vr, :vc] = (input1_valid > input2_valid)
    elif cmp_mode == "ge":
        mask_bits[:vr, :vc] = (input1_valid >= input2_valid)
    elif cmp_mode == "le":
        mask_bits[:vr, :vc] = (input1_valid <= input2_valid)

    # dst shape is same as src shape, but only first cols//8 columns store packed mask bytes
    # remaining columns are padding (zeros)
    # Use uint8 first to avoid overflow, then cast to int8
    golden = np.zeros(shape, dtype=np.uint8)
    
    # Pack mask bits: each byte stores 8 comparison results (1 bit each)
    packed_cols = vc // 8  # number of byte columns that store actual packed data
    
    for row in range(vr):
        for col_byte in range(packed_cols):
            byte_val = 0
            for bit in range(8):
                src_col = col_byte * 8 + bit
                if src_col < vc and mask_bits[row, src_col]:
                    byte_val |= (1 << bit)
            golden[row, col_byte] = byte_val

    # Cast to int8 for final output
    golden = golden.astype(dst_dtype)

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} cmp_mode={cmp_mode}")