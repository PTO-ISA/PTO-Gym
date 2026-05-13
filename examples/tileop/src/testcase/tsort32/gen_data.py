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
import os
import sys

# Add parent directory to path for st_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_common import setup_case_rng, save_case_data

from cases import CASES

BLOCK_SIZE = 32
FLOAT_DST_STRIDE_COEF = 2  # for f32
HALF_DST_STRIDE_COEF = 4   # for f16


def _to_tuple(shape):
    """Convert shape to tuple if needed."""
    if isinstance(shape, tuple):
        return shape
    return tuple(shape)


def get_stride_coef(dtype):
    """Get stride coefficient based on dtype."""
    if dtype == np.float16:
        return HALF_DST_STRIDE_COEF
    return FLOAT_DST_STRIDE_COEF


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    src_shape = _to_tuple(case["src_shape"])
    idx_shape = _to_tuple(case["idx_shape"])
    dst_shape = _to_tuple(case["dst_shape"])
    src_valid = _to_tuple(case["valid_shape"])
    idx_valid = _to_tuple(case["idx_vshape"])

    src_rows, src_cols = src_shape
    src_vr, src_vc = src_valid
    idx_vr, idx_vc = idx_valid

    # Generate random input data
    input_data = np.random.randint(1, 100, size=src_shape).astype(dtype)

    # Generate index data (0, 1, 2, ... for each row)
    # If idx_valid_rows == 1, same index is used for all rows
    if idx_vr == 1:
        idx_data = np.arange(src_cols, dtype=np.int32).reshape(1, src_cols)
    else:
        idx_data = np.arange(src_cols, dtype=np.int32).reshape(1, src_cols)
        idx_data = np.tile(idx_data, (src_rows, 1))

    # Compute golden: for each 32-element block, sort and output interleaved (value, index)
    # Output stride coef depends on dtype:
    #   - f32 uses stride_coef=2 (value+index pair occupies 2 f32 elements)
    #   - f16 uses stride_coef=4 (value occupies 1 f16, index stored as ui32 = 4 f16 positions)
    stride_coef = get_stride_coef(dtype)
    golden = np.zeros(dst_shape, dtype=dtype)

    for row in range(src_vr):
        for block_start in range(0, src_vc, BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, src_vc)
            block_size = block_end - block_start

            block_data = input_data[row, block_start:block_end].copy()
            block_idx = idx_data[0 if idx_vr == 1 else row, block_start:block_end].astype(np.int32)

            # For partial blocks, pad with NaN (negative NaN = max value) to make 32 elements
            if block_size < BLOCK_SIZE:
                # Use the same padding value as in tsort32_template.py
                # f16: 0x7C00 (+inf), bf16: 0x7FC0, f32: 0x7FC00000 (negative NaN)
                if dtype == np.float16:
                    pad_val = np.float16(0xFC00)  # +inf for f16
                elif hasattr(np, 'bfloat16') and dtype == np.bfloat16:
                    pad_val = np.bfloat16(0xFF80)
                else:
                    pad_val = np.float32(0xFF800000)  # negative NaN for f32

                # Pad block to 32 elements with +inf (will be sorted to end)
                padded_data = np.full(BLOCK_SIZE, pad_val, dtype=dtype)
                padded_data[:block_size] = block_data

                # Pad indices to 32 elements (indices for padding elements don't matter)
                padded_idx = np.zeros(BLOCK_SIZE, dtype=np.int32)
                padded_idx[:block_size] = block_idx

                # Sort the padded 32-element block in descending order
                # +inf values will be at the end after sorting
                sorted_indices = np.argsort(-padded_data)
                sorted_values = padded_data[sorted_indices]
                sorted_original_idx = padded_idx[sorted_indices]

                # Output interleaved (value, index) pairs for the full 32-element block
                # but only the first block_size elements are valid (padding elements at the end)
                dst_offset = block_start * stride_coef
                for i in range(BLOCK_SIZE):
                    golden[row, dst_offset + i * stride_coef] = sorted_values[i]
                    # Store index as int32 bit pattern
                    idx_u32 = np.array(sorted_original_idx[i], dtype=np.uint32)
                    if dtype == np.float16:
                        idx_bytes = idx_u32.tobytes()
                        golden[row, dst_offset + i * stride_coef + 1] = np.frombuffer(idx_bytes[:2], dtype=np.float16)[0]
                        golden[row, dst_offset + i * stride_coef + 2] = np.frombuffer(idx_bytes[2:], dtype=np.float16)[0]
                    else:
                        golden[row, dst_offset + i * stride_coef + 1] = idx_u32.view(np.float32)
            else:
                # Full 32-element block
                # Sort by value in descending order (largest to smallest)
                sorted_indices = np.argsort(-block_data)
                sorted_values = block_data[sorted_indices]
                sorted_original_idx = block_idx[sorted_indices]

                # Output interleaved (value, index) pairs with stride_coef
                dst_offset = block_start * stride_coef
                for i in range(BLOCK_SIZE):
                    golden[row, dst_offset + i * stride_coef] = sorted_values[i]
                    # Store index as int32 bit pattern
                    idx_u32 = np.array(sorted_original_idx[i], dtype=np.uint32)
                    if dtype == np.float16:
                        idx_bytes = idx_u32.tobytes()
                        golden[row, dst_offset + i * stride_coef + 1] = np.frombuffer(idx_bytes[:2], dtype=np.float16)[0]
                        golden[row, dst_offset + i * stride_coef + 2] = np.frombuffer(idx_bytes[2:], dtype=np.float16)[0]
                    else:
                        golden[row, dst_offset + i * stride_coef + 1] = idx_u32.view(np.float32)

    save_case_data(case["name"], {"input": input_data, "idx": idx_data.astype(np.uint32), "golden": golden})
    print(f"[INFO] gen_data: {case['name']} src_shape={src_shape} idx_shape={idx_shape} dst_shape={dst_shape} dtype={dtype.__name__}")