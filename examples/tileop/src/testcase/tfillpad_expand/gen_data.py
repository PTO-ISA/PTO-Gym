# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate golden data for tfillpad_expand test cases.

TFILLPAD_EXPAND semantics:
  1. Copy src.valid_shape data to dst
  2. Fill cols from src.valid_cols to dst.valid_cols with FillPadVal
  3. Fill rows from src.rows to dst.rows with FillPadVal

Note: LoadPadVal is used by TLOAD only, TFILLPAD_EXPAND uses FillPadVal for expansion.
"""

import os
import numpy as np
import struct

from cases import CASES, PADVAL_MAX, PADVAL_MIN, PADVAL_NEG1, PADVAL_ZERO


# FLT_MAX and -FLT_MAX (matching DSL PadValue.MAX/MIN)
def _float32_from_bits(bits: int) -> float:
    return struct.unpack(">f", bits.to_bytes(4, byteorder="big", signed=False))[0]

_FLT_MAX = _float32_from_bits(0x7F7FFFFF)  # ~3.4028235e+38
_FLT_MIN = _float32_from_bits(0xFF7FFFFF)  # ~-3.4028235e+38


def get_pad_value(dtype, padval_name):
    """Get the actual pad value for a dtype based on PadValue enum.

    Matches DSL PadValue.materialize_scalar behavior:
      - MAX: FLT_MAX for float (not inf), max for integers
      - MIN: -FLT_MAX for float (not -inf), min for integers
      - NEG1: -1.0 for float, -1 for integers
      - NULL/ZERO: 0
    """
    if padval_name == PADVAL_MAX:
        if np.issubdtype(dtype, np.floating):
            return np.float32(_FLT_MAX)
        else:
            return np.iinfo(dtype).max
    elif padval_name == PADVAL_MIN:
        if np.issubdtype(dtype, np.floating):
            return np.float32(_FLT_MIN)
        else:
            return np.iinfo(dtype).min
    elif padval_name == PADVAL_NEG1:
        if np.issubdtype(dtype, np.floating):
            return np.float32(-1.0)
        else:
            return dtype(-1)
    else:  # PADVAL_NULL or PADVAL_ZERO
        return dtype(0)


def setup_case_rng(case):
    """Set a per-case deterministic random seed."""
    np.random.seed(hash(case["name"]) & 0xFFFFFFFF)


def save_case_data(case_name, data_dict):
    """Create case directory and write {name}.bin for each entry."""
    os.makedirs(case_name, exist_ok=True)
    for name, arr in data_dict.items():
        arr.tofile(os.path.join(case_name, f"{name}.bin"))


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    src_shape = case["shape"]                # src physical (input size, matching tensor_view)
    src_valid = case["valid_shape"]          # src valid region (actual data in input)
    dst_shape = case["dst_shape"]            # dst physical
    dst_valid = case["dst_valid_shape"]      # dst valid (output size)
    fill_padval = case.get("fill_padval", PADVAL_ZERO)

    src_vr, src_vc = src_valid
    dst_vr, dst_vc = dst_valid

    # Generate input: random values in src valid region, zeros elsewhere
    # Input size = src_shape (matching tensor_view and C++ input)
    input_data = np.zeros(src_shape, dtype=dtype)
    input_data[:src_vr, :src_vc] = np.random.randint(1, 10, size=(src_vr, src_vc)).astype(dtype)

    # Generate golden: dst valid region (output size)
    golden = np.zeros(dst_valid, dtype=dtype)

    # Step 1: Copy src valid data to dst
    copy_vr = min(src_vr, dst_vr)
    copy_vc = min(src_vc, dst_vc)
    golden[:copy_vr, :copy_vc] = input_data[:copy_vr, :copy_vc]

    # Step 2: Fill column expansion region (cols from src_vc to dst_vc)
    if dst_vc > src_vc:
        fill_val = get_pad_value(dtype, fill_padval)
        golden[:dst_vr, src_vc:dst_vc] = fill_val

    # Step 3: Fill row expansion region (rows from src_vr to dst_vr)
    if dst_vr > src_vr:
        fill_val = get_pad_value(dtype, fill_padval)
        golden[src_vr:dst_vr, :dst_vc] = fill_val

    save_case_data(case["name"], {"input": input_data, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} src={src_shape} valid={src_valid} -> dst={dst_shape} "
          f"fill_pad={fill_padval} dtype={dtype.__name__}")