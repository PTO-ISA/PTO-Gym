#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tsort32 ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - src_shape:   (rows, cols) — allocated source tile dimensions.
  - idx_shape:   (rows, cols) — allocated index tile dimensions (can be 1 x cols for shared idx).
  - tmp_shape:   (rows, cols) — allocated tmp tile dimensions (optional, only for unaligned cases).
                 None for aligned cases (valid_cols % 32 == 0).
                 For unaligned cases: tmp_rows = 1, tmp_cols = ceil(valid_cols, 32).
  - dst_shape:   (rows, cols) — allocated destination tile dimensions.
                 For f32: dst_cols = src_cols * 4 (buffer allocation, but valid region is src_cols * 2).
                 For f16: dst_cols = src_cols * 2.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
                 For aligned cases: valid_cols must be multiple of 32 (BLOCK_SIZE).
                 For unaligned cases: valid_cols can be any value (requires tmp).
  - idx_vshape:  (idx_valid_rows, idx_valid_cols) — idx valid region.
                 If idx_valid_rows == 1, same idx is used for all rows.
  - dst_vshape:  (dst_valid_rows, dst_valid_cols) — dst valid region.
                 For f32: dst_vcols = src_vcols * 2 (stride coef = 2, interleaved value+index).
  - eps:         tolerance for numpy.allclose (atol and rtol).

tsort32 semantics:
  - Sorts data in 32-element blocks using vbitsort.
  - Output format: interleaved (sorted_value, original_index) pairs with stride coef = 2.
  - For each 32-element block, the output contains sorted values and their original indices.
  - Each pair occupies 2 element positions: [value0, idx0, value1, idx1, ...]

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    # f32 cases - basic shapes (aligned, no tmp needed)
    {
        "name": "f32_1x32",
        "dtype": np.float32,
        "src_shape": (1, 32),
        "idx_shape": (1, 32),
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (1, 128),      # buffer allocation (src_cols * 4)
        "valid_shape": (1, 32),
        "idx_vshape": (1, 32),
        "dst_vshape": (1, 64),      # actual valid output: src_cols * stride_coef = 32 * 2
        "eps": 1e-6,
    },
    {
        "name": "f32_1x64",
        "dtype": np.float32,
        "src_shape": (1, 64),
        "idx_shape": (1, 64),
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (1, 256),      # buffer allocation (src_cols * 4)
        "valid_shape": (1, 64),
        "idx_vshape": (1, 64),
        "dst_vshape": (1, 128),     # actual valid output: src_cols * stride_coef = 64 * 2
        "eps": 1e-6,
    },
    # f32 cases - multiple rows (aligned, no tmp needed)
    {
        "name": "f32_2x32",
        "dtype": np.float32,
        "src_shape": (2, 32),
        "idx_shape": (2, 32),
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (2, 128),      # buffer allocation (src_cols * 4)
        "valid_shape": (2, 32),
        "idx_vshape": (2, 32),
        "dst_vshape": (2, 64),      # actual valid output: src_cols * stride_coef = 32 * 2
        "eps": 1e-6,
    },
    {
        "name": "f32_16x32",
        "dtype": np.float32,
        "src_shape": (16, 32),
        "idx_shape": (16, 32),
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (16, 128),     # buffer allocation (src_cols * 4)
        "valid_shape": (16, 32),
        "idx_vshape": (16, 32),
        "dst_vshape": (16, 64),     # actual valid output: src_cols * stride_coef = 32 * 2
        "eps": 1e-6,
    },
    # f32 cases - shared idx (aligned, no tmp needed)
    {
        "name": "f32_2x64_shared_idx",
        "dtype": np.float32,
        "src_shape": (2, 64),
        "idx_shape": (1, 64),       # shared idx for all rows
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (2, 256),      # buffer allocation (src_cols * 4)
        "valid_shape": (2, 64),
        "idx_vshape": (1, 64),      # idx_valid_rows = 1 means shared idx
        "dst_vshape": (2, 128),     # actual valid output: src_cols * stride_coef = 64 * 2
        "eps": 1e-6,
    },
    {
        "name": "f32_16x64_shared_idx",
        "dtype": np.float32,
        "src_shape": (16, 64),
        "idx_shape": (1, 64),       # shared idx for all rows
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (16, 256),     # buffer allocation (src_cols * 4)
        "valid_shape": (16, 64),
        "idx_vshape": (1, 64),      # idx_valid_rows = 1 means shared idx
        "dst_vshape": (16, 128),    # actual valid output: src_cols * stride_coef = 64 * 2
        "eps": 1e-6,
    },
    # f32 cases - large shape (multiple vbitsort calls, aligned, no tmp needed)
    {
        "name": "f32_1x8192",
        "dtype": np.float32,
        "src_shape": (1, 8192),     # 256 * 32, requires loop_num > 1
        "idx_shape": (1, 8192),
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (1, 32768),    # buffer allocation (src_cols * 4)
        "valid_shape": (1, 8192),
        "idx_vshape": (1, 8192),
        "dst_vshape": (1, 16384),   # actual valid output: src_cols * stride_coef = 8192 * 2
        "eps": 1e-6,
    },
    # f32 cases - non-32-aligned (requires tmp buffer for padding)
    # Case 4 from C++: VALID_C=13, requires padding to 32-element block
    {
        "name": "f32_2x13",
        "dtype": np.float32,
        "src_shape": (2, 16),          # ALIGN_C = ceil(13*4, 32) / 4 = 16
        "idx_shape": (2, 16),
        "tmp_shape": (1, 16),          # unaligned: tmp_cols = ceil(13, 32) = 16
        "dst_shape": (2, 64),          # 4 * ALIGN_C = 64
        "valid_shape": (2, 13),        # non-32-aligned
        "idx_vshape": (2, 13),
        "dst_vshape": (2, 26),         # VALID_C * stride_coef = 13 * 2
        "eps": 1e-6,
    },
    # Case 5 from C++: VALID_C=4164, large non-aligned shape
    {
        "name": "f32_1x4164",
        "dtype": np.float32,
        "src_shape": (1, 8192),        # ALIGN_C = 8192 (from C++ hardcoded)
        "idx_shape": (1, 8192),
        "tmp_shape": (1, 4168),        # unaligned: tmp_cols = ceil(4164, 32) = 4168
        "dst_shape": (1, 32768),       # 4 * ALIGN_C = 32768
        "valid_shape": (1, 4164),      # non-32-aligned
        "idx_vshape": (1, 4164),
        "dst_vshape": (1, 8328),       # VALID_C * stride_coef = 4164 * 2
        "eps": 1e-6,
    },
    # Case 6 from C++: VALID_C=2084, multi-row non-aligned shape
    {
        "name": "f32_2x2084",
        "dtype": np.float32,
        "src_shape": (2, 3072),        # ALIGN_C = 3072 (from C++ hardcoded)
        "idx_shape": (2, 3072),
        "tmp_shape": (1, 2088),        # unaligned: tmp_cols = ceil(2084, 32) = 2088
        "dst_shape": (2, 12288),       # 4 * ALIGN_C = 12288
        "valid_shape": (2, 2084),      # non-32-aligned
        "idx_vshape": (2, 2084),
        "dst_vshape": (2, 4168),       # VALID_C * stride_coef = 2084 * 2
        "eps": 1e-6,
    },
    # f16 cases - basic shapes (aligned, no tmp needed)
    {
        "name": "f16_1x32",
        "dtype": np.float16,
        "src_shape": (1, 32),
        "idx_shape": (1, 32),
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (1, 128),      # buffer allocation (src_cols * 4 for f16)
        "valid_shape": (1, 32),
        "idx_vshape": (1, 32),
        "dst_vshape": (1, 128),     # actual valid output: src_cols * stride_coef = 32 * 4
        "eps": 1e-3,
    },
    {
        "name": "f16_4x64",
        "dtype": np.float16,
        "src_shape": (4, 64),
        "idx_shape": (4, 64),
        "tmp_shape": None,          # aligned: valid_cols % 32 == 0, no tmp
        "dst_shape": (4, 256),      # buffer allocation (src_cols * 4 for f16)
        "valid_shape": (4, 64),
        "idx_vshape": (4, 64),
        "dst_vshape": (4, 256),     # actual valid output: src_cols * stride_coef = 64 * 4
        "eps": 1e-3,
    },
]