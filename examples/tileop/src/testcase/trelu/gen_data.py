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

    if dtype == np.int32:
        input1 = np.random.randint(-3_000_000, 3_000_000, size=shape).astype(dtype)
    else:
        input1 = np.random.uniform(-10, 10, size=shape).astype(dtype)

    golden = np.maximum(input1, 0)

    save_case_data(case["name"], {"input": input1, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={case['valid_shape']} dtype={dtype.__name__}")