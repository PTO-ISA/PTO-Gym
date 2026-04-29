#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate golden data for texpands test cases."""

import os
import numpy as np

from cases import CASES


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
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    scalar = case["scalar"]

    # Convert scalar to the correct dtype
    scalar_val = dtype(scalar)

    # Generate golden: fill valid_shape region with scalar value
    golden = np.zeros(shape, dtype=dtype)
    vr, vc = valid_shape
    golden[:vr, :vc] = scalar_val

    save_case_data(case["name"], {"golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} scalar={scalar} dtype={dtype.__name__}")