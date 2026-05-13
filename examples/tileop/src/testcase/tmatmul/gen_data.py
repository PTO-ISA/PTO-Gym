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
from st_common import setup_case_rng, save_case_data


for case in CASES:
    setup_case_rng(case)

    shape_a = case["shape_a"]
    shape_b = case["shape_b"]
    dtype = case["dtype"]

    lhs = np.random.uniform(-1.0, 1.0, size=shape_a).astype(dtype)
    rhs = np.random.uniform(-1.0, 1.0, size=shape_b).astype(dtype)
    golden = np.matmul(lhs.astype(np.float32), rhs.astype(np.float32)).astype(np.float32)

    save_case_data(case["name"], {"input1": lhs, "input2": rhs, "golden": golden})
    print(
        f"[INFO] gen_data: {case['name']} "
        f"lhs={shape_a} rhs={shape_b} out={case['shape_c']} dtype={dtype.__name__}"
    )
