# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tfillpad_expand ST test cases.

Matches C++ reference test cases: Cases 8, 9

C++ expand mode parameters:
  - shape3: src physical rows
  - shape4: src physical cols
  - kTRows_: dst physical rows
  - kTCols_: dst physical cols
  - expand=true: TFILLPAD_EXPAND copies src valid data, fills expansion with FillPadVal

Case 8: runTFILLPAD<uint16_t, 1,1,1, 259,7, 260,32, 1, PadValue::Min, PadValue::Max, false, true>
Case 9: runTFILLPAD<int8_t, 1,1,1, 259,7, 260,64, 1, PadValue::Min, PadValue::Max, false, true>

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype
  - shape:       (rows, cols) — src tile physical dimensions (input size)
  - valid_shape: (valid_rows, valid_cols) — src valid region
  - dst_shape:   (rows, cols) — dst tile physical dimensions
  - dst_valid_shape: (valid_rows, valid_cols) — dst valid region (output size)
  - load_padval: PadValue for TLOAD (fill invalid columns in src tile)
  - fill_padval: PadValue for TFILLPAD_EXPAND (fill expansion region in dst)
  - eps:         tolerance for numpy.allclose
"""

import numpy as np

# PadValue enum values matching C++ definition
PADVAL_MAX = "Max"       # FLT_MAX for float, MAX for integers
PADVAL_MIN = "Min"       # -FLT_MAX for float, MIN for integers
PADVAL_NULL = "Null"     # no fill
PADVAL_ZERO = "Zero"     # zero fill
PADVAL_NEG1 = "Neg1"     # -1.0f for float, -1 for integers (Custom)

CASES = [
    # ========== Case 1: uint16, src=259x7, dst=260x32, expand, LoadPad=Min, FillPad=Max ==========

    {
        "name": "u16_260x32_src_259x7",
        "dtype": np.uint16,
        "shape": (259, 7),               # src physical (C++ shape3=259, shape4=7)
        "valid_shape": (259, 7),         # src valid region (actual data)
        "dst_shape": (260, 32),          # dst physical
        "dst_valid_shape": (260, 32),    # dst valid (output size)
        "load_padval": PADVAL_MIN,       # TLOAD: fill cols 7-31 with MIN (uint16 MIN=0)
        "fill_padval": PADVAL_MAX,       # TFILLPAD_EXPAND: fill expansion region with MAX (uint16 MAX=65535)
        "eps": 0,
    },

    # ========== Case 2: int8, src=259x7, dst=260x64, expand, LoadPad=Min, FillPad=Max ==========

    {
        "name": "s8_260x64_src_259x7",
        "dtype": np.int8,
        "shape": (259, 7),               # src physical (C++ shape3=259, shape4=7)
        "valid_shape": (259, 7),         # src valid region (actual data)
        "dst_shape": (260, 64),          # dst physical
        "dst_valid_shape": (260, 64),    # dst valid (output size)
        "load_padval": PADVAL_MIN,       # TLOAD: fill cols 7-63 with MIN (int8 MIN=-128)
        "fill_padval": PADVAL_MAX,       # TFILLPAD_EXPAND: fill expansion region with MAX (127)
        "eps": 0,
    },
]