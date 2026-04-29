#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tsels ST test cases.

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype for data (src/dst)
  - dtype_mask:  numpy dtype for mask
  - dst_shape:   (dst_rows, dst_cols) — allocated dst tile dimensions
  - mask_shape:  (mask_rows, mask_cols) — allocated mask tile dimensions
  - src_shape:   (src_rows, src_cols) — allocated src tile dimensions
  - valid_shape: (valid_rows, valid_cols) — effective computation region
  - eps:         tolerance for numpy.allclose (atol and rtol)
"""

import numpy as np

CASES = [
    {"name": "uint8_uint8_2x32_2x32_2x32_2x32", "dtype": np.uint8, "dtype_mask": np.uint8, "shape": (2, 32), "dst_shape": (2, 32), "dst_valid_shape": (2, 32), "mask_shape": (2, 32), "src_shape": (2, 32), "valid_shape": (2, 32), "eps": 0},
    {"name": "uint8_uint16_2x32_2x16_2x32_2x32", "dtype": np.uint8, "dtype_mask": np.uint16, "shape": (2, 32), "dst_shape": (2, 32), "dst_valid_shape": (2, 32), "mask_shape": (2, 16), "src_shape": (2, 32), "valid_shape": (2, 32), "eps": 0},
    {"name": "uint8_uint32_2x32_2x8_2x32_2x32", "dtype": np.uint8, "dtype_mask": np.uint32, "shape": (2, 32), "dst_shape": (2, 32), "dst_valid_shape": (2, 32), "mask_shape": (2, 8), "src_shape": (2, 32), "valid_shape": (2, 32), "eps": 0},
    {"name": "uint16_uint8_2x16_2x32_2x16_2x16", "dtype": np.uint16, "dtype_mask": np.uint8, "shape": (2, 16), "dst_shape": (2, 16), "dst_valid_shape": (2, 16), "mask_shape": (2, 32), "src_shape": (2, 16), "valid_shape": (2, 16), "eps": 0},
    {"name": "uint16_uint16_2x16_2x16_2x16_2x16", "dtype": np.uint16, "dtype_mask": np.uint16, "shape": (2, 16), "dst_shape": (2, 16), "dst_valid_shape": (2, 16), "mask_shape": (2, 16), "src_shape": (2, 16), "valid_shape": (2, 16), "eps": 0},
    {"name": "uint16_uint32_2x16_2x8_2x16_2x16", "dtype": np.uint16, "dtype_mask": np.uint32, "shape": (2, 16), "dst_shape": (2, 16), "dst_valid_shape": (2, 16), "mask_shape": (2, 8), "src_shape": (2, 16), "valid_shape": (2, 16), "eps": 0},
    {"name": "uint32_uint8_2x8_2x32_2x8_2x8", "dtype": np.uint32, "dtype_mask": np.uint8, "shape": (2, 8), "dst_shape": (2, 8), "dst_valid_shape": (2, 8), "mask_shape": (2, 32), "src_shape": (2, 8), "valid_shape": (2, 8), "eps": 0},
    {"name": "uint32_uint16_2x8_2x16_2x8_2x8", "dtype": np.uint32, "dtype_mask": np.uint16, "shape": (2, 8), "dst_shape": (2, 8), "dst_valid_shape": (2, 8), "mask_shape": (2, 16), "src_shape": (2, 8), "valid_shape": (2, 8), "eps": 0},
    {"name": "uint32_uint32_2x8_2x8_2x8_2x8", "dtype": np.uint32, "dtype_mask": np.uint32, "shape": (2, 8), "dst_shape": (2, 8), "dst_valid_shape": (2, 8), "mask_shape": (2, 8), "src_shape": (2, 8), "valid_shape": (2, 8), "eps": 0},
    {"name": "f16_uint8_2x16_2x32_2x16_2x16", "dtype": np.float16, "dtype_mask": np.uint8, "shape": (2, 16), "dst_shape": (2, 16), "dst_valid_shape": (2, 16), "mask_shape": (2, 32), "src_shape": (2, 16), "valid_shape": (2, 16), "eps": 1e-3},
    {"name": "f16_uint16_2x16_2x16_2x16_2x16", "dtype": np.float16, "dtype_mask": np.uint16, "shape": (2, 16), "dst_shape": (2, 16), "dst_valid_shape": (2, 16), "mask_shape": (2, 16), "src_shape": (2, 16), "valid_shape": (2, 16), "eps": 1e-3},
    {"name": "f16_uint32_2x16_2x8_2x16_2x16", "dtype": np.float16, "dtype_mask": np.uint32, "shape": (2, 16), "dst_shape": (2, 16), "dst_valid_shape": (2, 16), "mask_shape": (2, 8), "src_shape": (2, 16), "valid_shape": (2, 16), "eps": 1e-3},
    {"name": "f32_uint8_2x8_2x32_2x8_2x8", "dtype": np.float32, "dtype_mask": np.uint8, "shape": (2, 8), "dst_shape": (2, 8), "dst_valid_shape": (2, 8), "mask_shape": (2, 32), "src_shape": (2, 8), "valid_shape": (2, 8), "eps": 1e-6},
    {"name": "f32_uint16_2x8_2x16_2x8_2x8", "dtype": np.float32, "dtype_mask": np.uint16, "shape": (2, 8), "dst_shape": (2, 8), "dst_valid_shape": (2, 8), "mask_shape": (2, 16), "src_shape": (2, 8), "valid_shape": (2, 8), "eps": 1e-6},
    {"name": "f32_uint32_2x8_2x8_2x8_2x8", "dtype": np.float32, "dtype_mask": np.uint32, "shape": (2, 8), "dst_shape": (2, 8), "dst_valid_shape": (2, 8), "mask_shape": (2, 8), "src_shape": (2, 8), "valid_shape": (2, 8), "eps": 1e-6},
    {"name": "uint8_uint8_2x32_2x64_2x128_2x31", "dtype": np.uint8, "dtype_mask": np.uint8, "shape": (2, 32), "dst_shape": (2, 32), "dst_valid_shape": (2, 31), "mask_shape": (2, 64), "src_shape": (2, 128), "valid_shape": (2, 31), "eps": 0},
    {"name": "uint16_uint8_2x32_2x64_2x128_2x31", "dtype": np.uint16, "dtype_mask": np.uint8, "shape": (2, 32), "dst_shape": (2, 32), "dst_valid_shape": (2, 31), "mask_shape": (2, 64), "src_shape": (2, 128), "valid_shape": (2, 31), "eps": 0},
    {"name": "f32_uint8_2x32_2x64_2x128_2x31", "dtype": np.float32, "dtype_mask": np.uint8, "shape": (2, 32), "dst_shape": (2, 32), "dst_valid_shape": (2, 31), "mask_shape": (2, 64), "src_shape": (2, 128), "valid_shape": (2, 31), "eps": 1e-6},
    {"name": "uint8_uint8_32x672_32x96_32x672_32x666", "dtype": np.uint8, "dtype_mask": np.uint8, "shape": (32, 672), "dst_shape": (32, 672), "dst_valid_shape": (32, 666), "mask_shape": (32, 96), "src_shape": (32, 672), "valid_shape": (32, 666), "eps": 0},
    {"name": "f16_uint8_32x672_32x96_32x672_32x666", "dtype": np.float16, "dtype_mask": np.uint8, "shape": (32, 672), "dst_shape": (32, 672), "dst_valid_shape": (32, 666), "mask_shape": (32, 96), "src_shape": (32, 672), "valid_shape": (32, 666), "eps": 1e-3},
    {"name": "f32_uint8_32x672_32x96_32x672_32x666", "dtype": np.float32, "dtype_mask": np.uint8, "shape": (32, 672), "dst_shape": (32, 672), "dst_valid_shape": (32, 666), "mask_shape": (32, 96), "src_shape": (32, 672), "valid_shape": (32, 666), "eps": 1e-6},
    {"name": "f32_uint8_1x8192_1x4096_1x8192_1x8192", "dtype": np.float32, "dtype_mask": np.uint8, "shape": (1, 8192), "dst_shape": (1, 8192), "dst_valid_shape": (1, 8192), "mask_shape": (1, 4096), "src_shape": (1, 8192), "valid_shape": (1, 8192), "eps": 1e-6},
]