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
    dst_dtype = case["dst_dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]

    if dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        dtype_info = np.iinfo(dtype)
        input1 = np.random.randint(dtype_info.min, dtype_info.max, size=shape).astype(dtype)
    else:
        dtype_info = np.finfo(dtype)
        input1 = np.random.uniform(low=dtype_info.min, high=dtype_info.max, size=shape).astype(dtype)

    out_shape = (valid_shape[0], 1)
    golden = np.zeros(out_shape, dtype=dst_dtype)
    golden[:, 0:1] = np.argmax(input1[:, :valid_shape[1]], axis=1, keepdims=True).astype(dst_dtype)

    save_case_data(case["name"], {"input1": input1, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__}")
