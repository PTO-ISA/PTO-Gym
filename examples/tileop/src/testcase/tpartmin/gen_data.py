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

    # tpartmin semantics (based on pto-isa TPartBinOps.hpp TCopyPadOp):
    # Algorithm:
    # 1. dst[:] = Max (padding for min operation)
    # 2. dst[0:src0_vr, 0:src0_vc] = src0[0:src0_vr, 0:src0_vc] (copy src0 to dst)
    # 3. dst[0:src1_vr, 0:src1_vc] = min(dst[0:src1_vr, 0:src1_vc], src1[0:src1_vr, 0:src1_vc])
    #    (apply min in src1 valid region)

    src0_eq_dst = (src0_vr == dst_vr and src0_vc == dst_vc)
    src1_eq_dst = (src1_vr == dst_vr and src1_vc == dst_vc)

    if src0_eq_dst and src1_eq_dst:
        # Full min: both src0 and src1 cover entire dst
        golden[:dst_vr, :dst_vc] = np.minimum(input1[:dst_vr, :dst_vc], input2[:dst_vr, :dst_vc]).astype(dtype, copy=False)
    elif src0_eq_dst:
        # src0 covers dst, src1 is partial
        # dst = src0 (copy), then min(dst, src1) in src1 region = min(src0, src1) in src1 region, src0 in rest
        golden[:src1_vr, :src1_vc] = np.minimum(input1[:src1_vr, :src1_vc], input2[:src1_vr, :src1_vc]).astype(dtype, copy=False)
        if src1_vc < dst_vc:
            golden[:src1_vr, src1_vc:dst_vc] = input1[:src1_vr, src1_vc:dst_vc].copy()
        if src1_vr < dst_vr:
            golden[src1_vr:dst_vr, :dst_vc] = input1[src1_vr:dst_vr, :dst_vc].copy()
    elif src1_eq_dst:
        # src1 covers dst, src0 is partial
        # dst = Max, then copy src0 in src0 region, then min(dst, src1) in src1 region
        golden[:src0_vr, :src0_vc] = np.minimum(input1[:src0_vr, :src0_vc], input2[:src0_vr, :src0_vc]).astype(dtype, copy=False)
        if src0_vc < dst_vc:
            golden[:src0_vr, src0_vc:dst_vc] = input2[:src0_vr, src0_vc:dst_vc].copy()
        if src0_vr < dst_vr:
            golden[src0_vr:dst_vr, :dst_vc] = input2[src0_vr:dst_vr, :dst_vc].copy()
    else:
        min_vr = min(src0_vr, src1_vr)
        min_vc = min(src0_vc, src1_vc)

        # Region 1: [0:min_vr, 0:min_vc] - overlapping region (both src0 and src1 valid)
        golden[:min_vr, :min_vc] = np.minimum(input1[:min_vr, :min_vc], input2[:min_vr, :min_vc]).astype(dtype, copy=False)

        # Region 2: [0:src0_vr, min_vc:src0_vc] if src0_vc > min_vc
        if src0_vc > min_vc:
            golden[:src0_vr, min_vc:src0_vc] = input1[:src0_vr, min_vc:src0_vc].copy()

        # Region 3: [min_vr:src1_vr, 0:min_vc] if src1_vr > min_vr
        if src1_vr > min_vr:
            golden[min_vr:src1_vr, :min_vc] = input2[min_vr:src1_vr, :min_vc].copy()

        # Region 4: [min_vr:src1_vr, min_vc:src1_vc] if src1_vr > min_vr AND src1_vc > min_vc
        if src1_vr > min_vr and src1_vc > min_vc:
            golden[min_vr:src1_vr, min_vc:src1_vc] = input2[min_vr:src1_vr, min_vc:src1_vc].copy()

        # Region 5: [0:min_vr, src1_vc:src0_vc] if src0_vc > src1_vc
        if src0_vc > src1_vc and min_vr > 0:
            # Already handled in Region 2 if rows are [0:src0_vr]
            pass  # Region 2 covers this

        if src1_vr > src0_vr and src0_vc > src1_vc:
            # Region [src0_vr:src1_vr, src1_vc:src0_vc] = Max (neither covers)
            # This is correct for tpartmin - padding value is Max
            # For floats, we use np.inf. For integers, use dtype max.
            if dtype == np.float32:
                max_val = np.finfo(np.float32).max
            elif dtype == np.float16:
                max_val = np.finfo(np.float16).max
            elif dtype == np.int8:
                max_val = np.iinfo(np.int8).max
            elif dtype == np.uint8:
                max_val = np.iinfo(np.uint8).max
            elif dtype == np.int16:
                max_val = np.iinfo(np.int16).max
            elif dtype == np.uint16:
                max_val = np.iinfo(np.uint16).max
            elif dtype == np.int32:
                max_val = np.iinfo(np.int32).max
            elif dtype == np.uint32:
                max_val = np.iinfo(np.uint32).max
            else:
                max_val = np.iinfo(dtype).max
            golden[src0_vr:src1_vr, src1_vc:src0_vc] = max_val

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} src0_valid={src0_valid} src1_valid={src1_valid} dst_valid={dst_valid} dtype={dtype.__name__}")