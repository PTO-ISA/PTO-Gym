#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from cases import CASES
from st_common import save_case_data, validate_cases


validate_cases(CASES)

for case in CASES:
    rows = int(case["rows"])
    cols = int(case["cols"])
    seq = int(case["seq"])
    seed = int(case["seed"])

    rng = np.random.default_rng(seed)
    oldmax = rng.uniform(-3.0, 1.5, size=(rows,)).astype(np.float32)
    oldsum = rng.uniform(0.5, 4.0, size=(rows,)).astype(np.float32)
    qk = rng.normal(loc=0.0, scale=1.5, size=(rows, cols)).astype(np.float32)

    qk_active = qk[:, :seq]
    qk_rowmax = np.max(qk_active, axis=1)
    newmax = np.maximum(qk_rowmax, oldmax)
    tmp_active = np.exp(qk_active - newmax[:, None], dtype=np.float32)
    cursum = np.sum(tmp_active, axis=1, dtype=np.float32)
    raw_expmax = np.exp(oldmax - newmax, dtype=np.float32)
    newsum = raw_expmax * oldsum + cursum
    expmax = (raw_expmax * oldsum) / newsum
    out = np.zeros((rows, cols), dtype=np.float32)
    out[:, :seq] = tmp_active / newsum[:, None]

    zeros_state = np.zeros((rows,), dtype=np.float32)
    zeros_out = np.zeros((rows, cols), dtype=np.float32)

    save_case_data(
        case["name"],
        {
            "v1": oldmax,
            "v2": oldsum,
            "v3": qk.reshape(-1),
            "v4": zeros_state,
            "v5": zeros_state,
            "v6": zeros_state,
            "v7": zeros_out.reshape(-1),
            "v8": np.array([seq], dtype=np.int32),
            "v9": np.array([rows], dtype=np.int32),
            "golden_v4": newmax,
            "golden_v5": newsum,
            "golden_v6": expmax,
            "golden_v7": out.reshape(-1),
        },
    )
    print(
        f"[INFO] gen_data: {case['name']} rows={rows} cols={cols} "
        f"seq={seq} dtype={case['dtype'].__name__}"
    )
