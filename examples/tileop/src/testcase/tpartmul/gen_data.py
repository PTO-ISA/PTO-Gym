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


def _to_tuple(shape):
    """Convert shape to tuple if needed."""
    if isinstance(shape, tuple):
        return shape
    return tuple(shape)


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = _to_tuple(case["shape"])
    src0_valid = _to_tuple(case["valid_shape"])
    src1_valid = _to_tuple(case["src1_vshape"])
    dst_valid = _to_tuple(case["dst_vshape"])

    rows, cols = shape
    src0_vr, src0_vc = src0_valid
    src1_vr, src1_vc = src1_valid
    dst_vr, dst_vc = dst_valid

    input1 = np.random.randint(1, 10, size=shape).astype(dtype)
    input2 = np.random.randint(1, 10, size=shape).astype(dtype)

    golden = np.zeros(shape, dtype=dtype)

    # Compute golden according to tpartmul semantics from template:
    # If src0_valid == dst_valid: use tpart_op with src0 as full operand
    #   - If src1 row_less: mul for src1 region, copy src0 for remaining rows
    #   - If src1 col_less: copy src0 full, then mul for overlapping region
    # If src1_valid == dst_valid: use tpart_op with src1 as full operand (swap src0/src1)

    src0_eq_dst = (src0_vr == dst_vr and src0_vc == dst_vc)
    src1_eq_dst = (src1_vr == dst_vr and src1_vc == dst_vc)

    if src0_eq_dst:
        # src0 is the full operand matching dst
        src1_row_lt_dst = (src1_vr < dst_vr and src1_vc == dst_vc)
        src1_col_lt_dst = (src1_vr <= dst_vr and src1_vc < dst_vc)

        if src1_eq_dst:
            # Full mul: dst[:] = src0[:] * src1[:]
            golden[:dst_vr, :dst_vc] = (input1[:dst_vr, :dst_vc] * input2[:dst_vr, :dst_vc]).astype(dtype, copy=False)
        elif src1_col_lt_dst:
            # Col_less: first copy src0, then mul in overlapping region
            golden[:dst_vr, :dst_vc] = input1[:dst_vr, :dst_vc].copy()
            if src1_vc > 0:
                golden[:src1_vr, :src1_vc] = (input1[:src1_vr, :src1_vc] * input2[:src1_vr, :src1_vc]).astype(dtype, copy=False)
        elif src1_row_lt_dst:
            # Row_less: mul for src1 region, copy src0 for remaining rows
            if src1_vc > 0:
                golden[:src1_vr, :src1_vc] = (input1[:src1_vr, :src1_vc] * input2[:src1_vr, :src1_vc]).astype(dtype, copy=False)
            golden[src1_vr:dst_vr, :dst_vc] = input1[src1_vr:dst_vr, :dst_vc].copy()
    elif src1_eq_dst:
        # src1 is the full operand matching dst, swap src0/src1 in the logic
        src0_row_lt_dst = (src0_vr < dst_vr and src0_vc == dst_vc)
        src0_col_lt_dst = (src0_vr <= dst_vr and src0_vc < dst_vc)

        if src0_eq_dst:
            # Full mul: dst[:] = src0[:] * src1[:]
            golden[:dst_vr, :dst_vc] = (input1[:dst_vr, :dst_vc] * input2[:dst_vr, :dst_vc]).astype(dtype, copy=False)
        elif src0_col_lt_dst:
            # Col_less: first copy src1, then mul in overlapping region
            golden[:dst_vr, :dst_vc] = input2[:dst_vr, :dst_vc].copy()
            if src0_vc > 0:
                golden[:src0_vr, :src0_vc] = (input1[:src0_vr, :src0_vc] * input2[:src0_vr, :src0_vc]).astype(dtype, copy=False)
        elif src0_row_lt_dst:
            # Row_less: mul for src0 region, copy src1 for remaining rows
            if src0_vc > 0:
                golden[:src0_vr, :src0_vc] = (input1[:src0_vr, :src0_vc] * input2[:src0_vr, :src0_vc]).astype(dtype, copy=False)
            golden[src0_vr:dst_vr, :dst_vc] = input2[src0_vr:dst_vr, :dst_vc].copy()

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} src0_valid={src0_valid} src1_valid={src1_valid} dst_valid={dst_valid} dtype={dtype.__name__}")