# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tfillpad_inplace ST test cases.

Matches C++ reference test case: Case 5

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype
  - shape:       (rows, cols) — tile dimensions (physical buffer size)
  - valid_shape: (valid_rows, valid_cols) — valid region (smaller than shape)
  - eps:         tolerance for numpy.allclose
"""

import numpy as np

CASES = [
    # ========== Case: float, src_valid == dst_valid (no expansion) ==========

    {
        "name": "f32_260x16_noexpand",
        "dtype": np.float32,
        "src_shape": (260, 16),         # src physical
        "src_valid": (260, 16),         # src valid = dst valid (no expansion)
        "dst_shape": (260, 16),         # dst physical
        "dst_valid": (260, 16),         # dst valid = full output
        "fill_padval": "Max",           # FillPadVal (not used since no expansion)
        "eps": 1e-6,
    },
]