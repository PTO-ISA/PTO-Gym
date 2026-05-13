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
import struct
import math
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    high_precision = case["high_precision"]

    if high_precision:
        hex_threshold = '007FFFFF'
        bound_val = struct.unpack('!f', bytes.fromhex(hex_threshold))[0]
        max_val = math.log(bound_val)
        min_val = max_val * 2
        input = np.random.uniform(min_val, max_val, size=shape).astype(dtype)
    else:
        input = np.random.randn(*shape).astype(dtype)

    golden = np.zeros(shape, dtype=dtype)
    vr, vc = valid_shape
    golden[:vr, :vc] = np.exp(input[:vr, :vc]).astype(dtype, copy=False)

    save_case_data(case["name"], {"input": input, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__} high_precision={high_precision}")
