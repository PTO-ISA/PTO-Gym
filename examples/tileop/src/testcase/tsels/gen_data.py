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
    dtype_mask = case["dtype_mask"]
    dst_shape = case["dst_shape"]
    mask_shape = case["mask_shape"]
    src_shape = case["src_shape"]
    valid_shape = case["valid_shape"]
    height, width = valid_shape

    if dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        dtype_info = np.iinfo(dtype)
        input1 = np.random.randint(dtype_info.min, dtype_info.max, size=src_shape).astype(dtype)
        input2 = np.random.randint(dtype_info.min, dtype_info.max, size=[1]).astype(dtype)
    else:
        dtype_info = np.finfo(dtype)
        input1 = np.random.uniform(low=dtype_info.min, high=dtype_info.max, size=src_shape).astype(dtype)
        input2 = np.random.uniform(low=dtype_info.min, high=dtype_info.max, size=[1]).astype(dtype)

    mask_dtype_info = np.iinfo(dtype_mask)
    mask = np.random.randint(mask_dtype_info.min, mask_dtype_info.max, size=mask_shape).astype(dtype_mask)
    mask_u8view = mask.view(np.uint8).reshape(mask_shape[0], -1)
    golden = np.zeros(dst_shape, dtype=dtype)

    for y in range(height):
        for x in range(width):
            do_select = (1 << (x & 7)) & mask_u8view[y, x >> 3]
            golden[y, x] = input1[y, x] if do_select != 0 else input2[0]

    save_case_data(case["name"], {"mask": mask, "input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} dst={dst_shape} mask={mask_shape} src={src_shape} valid={valid_shape} dtype={dtype.__name__} mask_dtype={dtype_mask.__name__}")