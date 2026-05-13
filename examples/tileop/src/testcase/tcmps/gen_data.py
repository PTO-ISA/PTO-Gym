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

# Scalar value for comparison (matches the scalar passed in launch.cpp)
SCALAR = 5.0

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    out_dtype = case["out_dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]

    # Generate random input matching testcase/tcmps pattern
    if np.issubdtype(dtype, np.floating):
        input1 = np.random.randint(-5, 5, size=shape).astype(dtype)
    else:
        input1 = np.random.randint(1, 10, size=shape).astype(dtype)

    vr, vc = valid_shape
    if np.issubdtype(dtype, np.floating):
        scalar_val = dtype(SCALAR)
    else:
        scalar_val = dtype(int(SCALAR))

    # Compute element-wise comparison result (0 or 1 per element)
    # Using "lt" mode to match the template
    cmp_result = (input1[:vr, :vc] < scalar_val).astype(np.uint8, copy=False)

    # tcmps output uses psts:
    # - 32B: 64 elements -> 32 bytes (NORM mode, sequential, bit_pos = col_in_iter * 4)
    # - 16B: 128 elements -> 16 bytes (PK mode, bit_pos = col_in_iter)
    # - 8B: 256 elements -> 32 bytes (NORM mode, sequential, bit_pos = col_in_iter)
    elem_size = np.dtype(dtype).itemsize
    lanes = 256 // elem_size
    if elem_size == 4:  # 32B: 2 vcmps + dintlv_b8 -> PK mode (16 bytes per iteration)
        bytes_per_iter = 16
        bit_multiplier = 1
        # For 32B, each iteration processes 2 repeats (128 elements)
        # Element linear index maps to bit position after dintlv_b8
    elif elem_size == 2:  # 16B: PK mode (16 bytes per iteration)
        bytes_per_iter = 16
        bit_multiplier = 1
    else:  # 8B: NORM mode (32 bytes per iteration)
        bytes_per_iter = 32
        bit_multiplier = 1
    
    # Calculate iterations (total)
    total_elm = vr * vc
    if elem_size == 4:  # 32B: special handling for linear offset
        repeat_times = (total_elm + lanes - 1) // lanes + 1
        total_iters = repeat_times // 2
    else:
        iters_per_row = (vc + lanes - 1) // lanes

    total_elm = vr * vc
    if elem_size == 4:  # 32B: special handling for linear offset
        repeat_times = (total_elm + lanes - 1) // lanes + 1
        total_iters = repeat_times // 2
        total_output_bytes = total_iters * bytes_per_iter
    else:
        iters_per_row = (vc + lanes - 1) // lanes
        total_iters = vr * iters_per_row
        total_output_bytes = total_iters * bytes_per_iter
    
    # Output buffer size matches actual output
    golden = np.zeros(total_output_bytes, dtype=np.uint8)

    for row in range(vr):
        for col in range(vc):
            if cmp_result[row, col]:
                if elem_size == 4:  # 32B: PK mode after dintlv_b8 with linear offset
                    # Linear element index
                    linear_idx = row * vc + col
                    # Each iteration processes 128 elements (2 repeats of 64)
                    iter_idx = linear_idx // (2 * lanes)
                    # Position within the 128-element block
                    pos_in_block = linear_idx % (2 * lanes)
                    # PK mode: bit position = pos_in_block
                    bit_pos = pos_in_block
                    # Byte offset (linear)
                    byte_idx = iter_idx * bytes_per_iter + (bit_pos // 8)
                    bit_idx = bit_pos % 8
                else:  # 16B and 8B
                    col_in_iter = col % lanes
                    bit_pos = col_in_iter * bit_multiplier
                    byte_idx = (row * iters_per_row + col // lanes) * bytes_per_iter + (bit_pos // 8)
                    bit_idx = bit_pos % 8
                
                if byte_idx < total_output_bytes:
                    golden[byte_idx] |= (1 << bit_idx)

    save_case_data(case["name"], {"input1": input1, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__} out_dtype={out_dtype.__name__} scalar={SCALAR}")
