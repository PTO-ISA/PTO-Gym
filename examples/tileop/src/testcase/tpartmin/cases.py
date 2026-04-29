#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tpartmin ST test cases.

Each case defines:
  - name:            case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:           numpy dtype (e.g. np.float32).
  - shape:           (rows, cols) — allocated tile dimensions (same for src0/src1/dst).
  - valid_shape:     (valid_rows, valid_cols) — src0 valid region (src0_eq_dst scenario).
  - src1_vshape:     (src1_valid_rows, src1_valid_cols) — src1 valid region.
                     May be smaller than dst valid region for partial min cases.
  - dst_vshape:      (dst_valid_rows, dst_valid_cols) — dst valid region.
  - eps:             tolerance for numpy.allclose (atol and rtol).

tpartmin semantics:
  - If src0_valid == dst_valid: dst[:src1_rows,:src1_cols] = min(src0[:src1_rows,:src1_cols], src1[:src1_rows,:src1_cols])
                                dst[src1_rows:,:] = src0[src1_rows:,:] (copy remaining rows)
                                OR (for col_less) dst[:,:src1_cols] = min(src0[:,:src1_cols], src1[:,:src1_cols])
                                dst[:,src1_cols:] = src0[:,src1_cols:] (copy remaining cols)
  - If src1_valid == dst_valid: similar logic with src1 as the full operand.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    # float32 cases from pto-isa
    {
        "name": "f32_64x64_full",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),      # src0 valid region
        "src1_vshape": (64, 64),      # src1 valid region (same as dst)
        "dst_vshape": (64, 64),       # dst valid region
        "eps": 1e-6,
    },
    {
        "name": "f32_2x24_src1_col_less",
        "dtype": np.float32,
        "shape": (2, 24),
        "valid_shape": (2, 24),       # src0 valid region (equals dst)
        "src1_vshape": (2, 8),        # src1 valid region (col_less)
        "dst_vshape": (2, 24),        # dst valid region
        "eps": 1e-6,
    },
    {
        "name": "f32_128x64_src1_row_less",
        "dtype": np.float32,
        "shape": (128, 64),
        "valid_shape": (128, 64),     # src0 valid region (equals dst)
        "src1_vshape": (96, 64),      # src1 valid region (row_less)
        "dst_vshape": (128, 64),      # dst valid region
        "eps": 1e-6,
    },
    {
        "name": "f32_95x95_full",
        "dtype": np.float32,
        "shape": (95, 95),
        "valid_shape": (95, 95),      # src0 valid region
        "src1_vshape": (95, 95),      # src1 valid region (same as dst)
        "dst_vshape": (95, 95),       # dst valid region
        "eps": 1e-6,
    },
    {
        "name": "f32_122x123_complex",
        "dtype": np.float32,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region (src1 rows, src0 cols)
        "eps": 1e-6,
    },
    # float16 cases from pto-isa
    {
        "name": "f16_122x123_complex",
        "dtype": np.float16,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region
        "eps": 1e-3,
    },
    # int16 cases from pto-isa
    {
        "name": "i16_122x123_complex",
        "dtype": np.int16,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region
        "eps": 0,
    },
    # int32 cases from pto-isa
    {
        "name": "i32_122x123_complex",
        "dtype": np.int32,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region
        "eps": 0,
    },
    # uint16 cases from pto-isa
    {
        "name": "u16_122x123_complex",
        "dtype": np.uint16,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region
        "eps": 0,
    },
    # uint32 cases from pto-isa
    {
        "name": "u32_122x123_complex",
        "dtype": np.uint32,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region
        "eps": 0,
    },
    # int8 cases from pto-isa
    {
        "name": "i8_122x123_complex",
        "dtype": np.int8,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region
        "eps": 0,
    },
    # uint8 cases from pto-isa
    {
        "name": "u8_122x123_complex",
        "dtype": np.uint8,
        "shape": (122, 123),
        "valid_shape": (104, 123),    # src0 valid region
        "src1_vshape": (122, 110),    # src1 valid region
        "dst_vshape": (122, 123),     # dst valid region
        "eps": 0,
    },
]