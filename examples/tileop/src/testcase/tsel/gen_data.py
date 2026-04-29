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

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    vr, vc = valid_shape
    mask_cols = (vc + 7) // 8

    src0 = np.random.randint(1, 10, size=shape).astype(dtype)
    src1 = np.random.randint(1, 10, size=shape).astype(dtype)
    mask = np.random.randint(0, 256, size=(vr, mask_cols), dtype=np.uint8)

    golden = np.zeros(shape, dtype=dtype)
    src0_valid = src0[:vr, :vc]
    src1_valid = src1[:vr, :vc]
    for row in range(vr):
        for packed_col in range(mask_cols):
            byte = int(mask[row, packed_col])
            for bit in range(8):
                col = packed_col * 8 + bit
                if col >= vc:
                    break
                golden[row, col] = src0_valid[row, col] if ((byte >> bit) & 1) else src1_valid[row, col]

    save_case_data(case["name"], {"input1": src0, "input2": src1, "input3": mask, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__}")