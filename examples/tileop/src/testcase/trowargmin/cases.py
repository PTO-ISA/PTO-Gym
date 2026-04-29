#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trowargmin ST test cases — aligned with pto-isa."""

import numpy as np

CASES = [
    # uint32_dst + float32_src
    {
        "name": "uint32_float_8x1_8x8_8x8",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (8, 8),
        "valid_shape": (8, 8),
        "eps": 0,
    },
    {
        "name": "uint32_float_1024x1_1024x8_1024x8",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (1024, 8),
        "valid_shape": (1024, 8),
        "eps": 0,
    },
    {
        "name": "uint32_float_16x1_13x16_13x13",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (13, 16),
        "valid_shape": (13, 13),
        "eps": 0,
    },
    {
        "name": "uint32_float_1024x1_1023x24_1023x17",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (1023, 24),
        "valid_shape": (1023, 17),
        "eps": 0,
    },
    {
        "name": "uint32_float_8x1_8x64_8x64",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (8, 64),
        "valid_shape": (8, 64),
        "eps": 0,
    },
    {
        "name": "uint32_float_264x1_260x64_260x64",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (260, 64),
        "valid_shape": (260, 64),
        "eps": 0,
    },
    {
        "name": "uint32_float_8x1_1x128_1x128",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (1, 128),
        "valid_shape": (1, 128),
        "eps": 0,
    },
    {
        "name": "uint32_float_64x1_32x128_32x128",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (32, 128),
        "valid_shape": (32, 128),
        "eps": 0,
    },
    {
        "name": "uint32_float_8x1_3x4096_3x4095",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (3, 4096),
        "valid_shape": (3, 4095),
        "eps": 0,
    },
    {
        "name": "uint32_float_8x1_2x16384_2x16381",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (2, 16384),
        "valid_shape": (2, 16381),
        "eps": 0,
    },
    # uint32_dst + float16_src
    {
        "name": "uint32_half_16x1_2x16_2x16",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (2, 16),
        "valid_shape": (2, 16),
        "eps": 0,
    },
    {
        "name": "uint32_half_16x1_13x16_13x13",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (13, 16),
        "valid_shape": (13, 13),
        "eps": 0,
    },
    {
        "name": "uint32_half_272x1_260x64_260x64",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (260, 64),
        "valid_shape": (260, 64),
        "eps": 0,
    },
    {
        "name": "uint32_half_16x1_3x8192_3x8191",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (3, 8192),
        "valid_shape": (3, 8191),
        "eps": 0,
    },
    {
        "name": "uint32_half_16x1_1x16384_1x16381",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (1, 16384),
        "valid_shape": (1, 16381),
        "eps": 0,
    },
    {
        "name": "uint32_half_16x1_1x32768_1x32761",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (1, 32768),
        "valid_shape": (1, 32761),
        "eps": 0,
    },
    # int32_dst + float32_src
    {
        "name": "int32_float_16x1_13x16_13x13",
        "dtype": np.float32,
        "dst_dtype": np.int32,
        "shape": (13, 16),
        "valid_shape": (13, 13),
        "eps": 0,
    },
    # int32_dst + float16_src
    {
        "name": "int32_half_16x1_13x16_13x13",
        "dtype": np.float16,
        "dst_dtype": np.int32,
        "shape": (13, 16),
        "valid_shape": (13, 13),
        "eps": 0,
    },
    # uint32_dst + float32_src (dst col > 1)
    {
        "name": "uint32_float_3x8_3x3480_3x3473",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (3, 3480),
        "valid_shape": (3, 3473),
        "eps": 0,
    },
    {
        "name": "uint32_float_260x8_260x64_260x64",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (260, 64),
        "valid_shape": (260, 64),
        "eps": 0,
    },
    {
        "name": "uint32_float_1023x8_1023x24_1023x17",
        "dtype": np.float32,
        "dst_dtype": np.uint32,
        "shape": (1023, 24),
        "valid_shape": (1023, 17),
        "eps": 0,
    },
    # uint32_dst + float16_src (dst col > 1)
    {
        "name": "uint32_half_3x16_3x3488_3x3473",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (3, 3488),
        "valid_shape": (3, 3473),
        "eps": 0,
    },
    {
        "name": "uint32_half_260x16_260x64_260x64",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (260, 64),
        "valid_shape": (260, 64),
        "eps": 0,
    },
    {
        "name": "uint32_half_1023x16_1023x32_1023x17",
        "dtype": np.float16,
        "dst_dtype": np.uint32,
        "shape": (1023, 32),
        "valid_shape": (1023, 17),
        "eps": 0,
    },
]
