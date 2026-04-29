#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tfillpad ST test cases.

Matches C++ reference test cases exactly (Cases 1-13).

PadValue semantics:
  - Max: +inf for float, MAX for integers
  - Min: -inf for float, MIN for integers
  - Null: no fill (keep original value)
  - Custom(-1.0f): -1.0f for float, -1 for integers

Each case defines:
  - name:        case identifier (must match main.cpp kCases[] and launch.cpp)
  - dtype:       numpy dtype
  - shape:       (rows, cols) — dst tile physical dimensions
  - valid_shape: (valid_rows, valid_cols) — dst valid region (output size)
  - src_shape:   (rows, cols) — src tile physical dimensions (optional, default=dst)
  - src_valid_shape: (valid_rows, valid_cols) — src valid region (optional, default=dst_valid)
  - load_padval: PadValue for TLOAD (fill invalid columns in src tile)
  - fill_padval: PadValue for TFILLPAD (fill expansion region in dst)
  - eps:         tolerance for numpy.allclose
"""

import numpy as np

# PadValue enum values matching C++ definition
PADVAL_MAX = "Max"       # +inf for float, MAX for integers
PADVAL_MIN = "Min"       # -inf for float, MIN for integers
PADVAL_NULL = "Null"     # no fill (keep original value, treated as 0 in golden)
PADVAL_ZERO = "Zero"     # zero fill
PADVAL_NEG1 = "Neg1"     # -1.0f for float, -1 for integers (Custom)

CASES = [
    # ========== Case 1: float, 128x127 -> 128x128, PadMax ==========
    # C++: runTFILLPAD<float, 1,1,1, 128,127, 128,128, 1, PadValue::Max, PadValue::Max>

    {
        "name": "f32_128x128_pad_128x127",
        "dtype": np.float32,
        "shape": (128, 128),              # dst tile physical
        "valid_shape": (128, 128),        # dst valid (output size)
        "src_shape": (128, 127),          # src tile physical (127 cols, < dst 128)
        "src_valid_shape": (128, 127),    # src valid = full src
        "load_padval": PADVAL_MAX,        # TLOAD: fill col 127 with +inf
        "fill_padval": PADVAL_MAX,        # TFILLPAD: no expansion needed
        "eps": 1e-6,
    },

    # ========== Case 2: float, 128x127 -> 128x160, PadMax ==========
    # C++: runTFILLPAD<float, 1,1,1, 128,127, 128,160, 1, PadValue::Max, PadValue::Max>

    {
        "name": "f32_128x160_pad_128x127",
        "dtype": np.float32,
        "shape": (128, 160),              # dst tile physical
        "valid_shape": (128, 160),        # dst valid (output size)
        "src_shape": (128, 127),          # src tile physical
        "src_valid_shape": (128, 127),    # src valid
        "load_padval": PADVAL_MAX,        # TLOAD: fill col 127 with +inf
        "fill_padval": PADVAL_MAX,        # TFILLPAD: fill cols 128-159 with +inf
        "eps": 1e-6,
    },

    # ========== Case 3: float, 128x127 -> 128x160, LoadPad=Min, FillPad=Max ==========
    # C++: runTFILLPAD<float, 1,1,1, 128,127, 128,160, 1, PadValue::Min, PadValue::Max>

    {
        "name": "f32_128x160_pad_128x127_v2",
        "dtype": np.float32,
        "shape": (128, 160),              # dst tile physical
        "valid_shape": (128, 160),        # dst valid (output size)
        "src_shape": (128, 127),          # src tile physical
        "src_valid_shape": (128, 127),    # src valid
        "load_padval": PADVAL_MIN,        # TLOAD: fill col 127 with -inf
        "fill_padval": PADVAL_MAX,        # TFILLPAD: fill cols 128-159 with +inf
        "eps": 1e-6,
    },

    # ========== Case 4: float, 260x7 -> 260x16, PadMin/Max ==========
    # C++: runTFILLPAD<float, 1,1,1, 260,7, 260,16, 1, PadValue::Min, PadValue::Max>

    {
        "name": "f32_260x16_pad_260x7",
        "dtype": np.float32,
        "shape": (260, 16),               # dst tile physical
        "valid_shape": (260, 16),         # dst valid (output size)
        "src_shape": (260, 7),            # src tile physical
        "src_valid_shape": (260, 7),      # src valid
        "load_padval": PADVAL_MIN,        # TLOAD: fill cols 8-15 with -inf (32B aligned tile)
        "fill_padval": PADVAL_MAX,        # TFILLPAD: no expansion needed
        "eps": 1e-6,
    },

    # ========== Case 6: uint16, 260x7 -> 260x32, PadMin/Max ==========
    # C++: runTFILLPAD<uint16_t, 1,1,1, 260,7, 260,32, 1, PadValue::Min, PadValue::Max>

    {
        "name": "u16_260x32_pad_260x7",
        "dtype": np.uint16,
        "shape": (260, 32),               # dst tile physical
        "valid_shape": (260, 32),         # dst valid (output size)
        "src_shape": (260, 7),            # src tile physical
        "src_valid_shape": (260, 7),      # src valid
        "load_padval": PADVAL_MIN,        # TLOAD: fill cols 8-31 with MIN (uint16 0)
        "fill_padval": PADVAL_MAX,        # TFILLPAD: fill cols 8-31 with MAX (uint16 65535)
        "eps": 0,
    },

    # ========== Case 7: int8, 260x7 -> 260x64, PadMin/Max ==========
    # C++: runTFILLPAD<int8_t, 1,1,1, 260,7, 260,64, 1, PadValue::Min, PadValue::Max>

    {
        "name": "s8_260x64_pad_260x7",
        "dtype": np.int8,
        "shape": (260, 64),               # dst tile physical
        "valid_shape": (260, 64),         # dst valid (output size)
        "src_shape": (260, 7),            # src tile physical
        "src_valid_shape": (260, 7),      # src valid
        "load_padval": PADVAL_MIN,        # TLOAD: fill cols 8-63 with MIN (int8 -128)
        "fill_padval": PADVAL_MAX,        # TFILLPAD: no expansion needed
        "eps": 0,
    },

    # ========== Case 10: int16, 260x7 -> 260x32, PadMin/Min ==========
    # C++: runTFILLPAD<int16_t, 1,1,1, 260,7, 260,32, 1, PadValue::Min, PadValue::Min>

    {
        "name": "s16_260x32_pad_260x7",
        "dtype": np.int16,
        "shape": (260, 32),               # dst tile physical
        "valid_shape": (260, 32),         # dst valid (output size)
        "src_shape": (260, 7),            # src tile physical
        "src_valid_shape": (260, 7),      # src valid
        "load_padval": PADVAL_MIN,        # TLOAD: fill cols 8-31 with MIN (int16 -32768)
        "fill_padval": PADVAL_MIN,        # TFILLPAD: no expansion needed
        "eps": 0,
    },

    # ========== Case 11: int32, 260x7 -> 260x32, PadMin/Min ==========
    # C++: runTFILLPAD<int32_t, 1,1,1, 260,7, 260,32, 1, PadValue::Min, PadValue::Min>

    {
        "name": "s32_260x32_pad_260x7",
        "dtype": np.int32,
        "shape": (260, 32),               # dst tile physical
        "valid_shape": (260, 32),         # dst valid (output size)
        "src_shape": (260, 7),            # src tile physical
        "src_valid_shape": (260, 7),      # src valid
        "load_padval": PADVAL_MIN,        # TLOAD: fill cols 8-31 with MIN (int32 -2147483648)
        "fill_padval": PADVAL_MIN,        # TFILLPAD: no expansion needed
        "eps": 0,
    },

    # ========== Case 12: float, 128x64 -> 128x128, LoadPad=Null, FillPad=Neg1 ==========
    # C++: runTFILLPAD<float, 1,1,1, 128,64, 128,128, 1, PadValue::Null, PadCustomNeg1>

    {
        "name": "f32_128x128_pad_128x64_neg1",
        "dtype": np.float32,
        "shape": (128, 128),              # dst tile physical
        "valid_shape": (128, 128),        # dst valid = full dst (output size)
        "src_shape": (128, 64),           # src tile physical (64 cols)
        "src_valid_shape": (128, 64),     # src valid = full src
        "load_padval": PADVAL_NULL,       # TLOAD: no fill (src cols 64 aligned to 32B)
        "fill_padval": PADVAL_NEG1,       # TFILLPAD: fill cols 64-127 with -1.0f
        "eps": 1e-6,
    },

    # ========== Case 13: float, 128x127 -> 128x160, LoadPad=Neg1, FillPad=Neg1 ==========
    # C++: runTFILLPAD<float, 1,1,1, 128,127, 128,160, 1, PadCustomNeg1, PadCustomNeg1>

    {
        "name": "f32_128x160_pad_128x127_neg1",
        "dtype": np.float32,
        "shape": (128, 160),              # dst tile physical
        "valid_shape": (128, 160),        # dst valid = full dst (output size) - CHANGED!
        "src_shape": (128, 127),          # src tile physical (127 cols)
        "src_valid_shape": (128, 127),    # src valid = full src
        "load_padval": PADVAL_NEG1,       # TLOAD: fill col 127 with -1.0f (127 not 32B aligned)
        "fill_padval": PADVAL_NEG1,       # TFILLPAD: fill cols 128-159 with -1.0f
        "eps": 1e-6,
    },
]