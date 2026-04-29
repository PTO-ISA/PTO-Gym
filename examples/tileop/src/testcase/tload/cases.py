#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

CASES = [
    {
        "name": "nd_f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
    {
        "name": "dn_f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
    {
        "name": "nz_f32_128x128",
        "dtype": np.float32,
        "shape": (128, 128),
        "valid_shape": (128, 128),
        "eps": 1e-6,
    },
    {
        "name": "nd_pad_zero_f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 63),
        "eps": 1e-6,
        "golden_fill": 0.0,
    },
    {
        "name": "dn_pad_max_f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (15, 64),
        "eps": 1e-6,
        "golden_fill": np.finfo(np.float32).max,
    },
    {
        "name": "nz_pad_min_f32_128x128",
        "dtype": np.float32,
        "shape": (128, 128),
        "valid_shape": (64, 128),
        "eps": 1e-6,
        "golden_fill": np.finfo(np.float32).min,
    },
]


def build_expected_output(case, input_arr):
    shape = case["shape"]
    vr, vc = case["valid_shape"]
    dtype = case["dtype"]

    if "golden_fill" in case:
        golden = np.full(shape, case["golden_fill"], dtype=dtype)
    else:
        golden = np.empty(shape, dtype=dtype)

    if case["name"].startswith("dn_pad_"):
        flat_in = np.asarray(input_arr, dtype=dtype).reshape(-1)
        flat_golden = golden.reshape(-1)
        physical_rows = shape[0]
        for col in range(vc):
            start = physical_rows * col
            flat_golden[start : start + vr] = flat_in[start : start + vr]
        return golden

    if case["name"].startswith("nz_pad_"):
        flat_in = np.asarray(input_arr, dtype=dtype).reshape(-1)
        flat_golden = golden.reshape(-1)
        block_rows = 8
        block_size = block_rows * shape[1]
        num_blocks = shape[0] // block_rows
        valid_rows_per_block = vr // num_blocks
        for block in range(num_blocks):
            base = block * block_size
            valid_elems = valid_rows_per_block * shape[1]
            flat_golden[base : base + valid_elems] = flat_in[base : base + valid_elems]
        return golden

    if "golden_fill" in case:
        golden[:vr, :vc] = input_arr[:vr, :vc]
        return golden

    return np.asarray(input_arr, dtype=dtype).copy()


def select_compared_region(case, arr):
    vr, vc = case["valid_shape"]

    if case["name"].startswith("dn_pad_"):
        flat = np.asarray(arr).reshape(-1)
        physical_rows = case["shape"][0]
        pieces = [flat[physical_rows * col : physical_rows * col + vr] for col in range(vc)]
        return np.concatenate(pieces) if pieces else flat[:0]

    if case["name"].startswith("nz_pad_"):
        flat = np.asarray(arr).reshape(-1)
        shape = case["shape"]
        block_rows = 8
        block_size = block_rows * shape[1]
        num_blocks = shape[0] // block_rows
        valid_rows_per_block = vr // num_blocks
        pieces = []
        for block in range(num_blocks):
            base = block * block_size
            valid_elems = valid_rows_per_block * shape[1]
            pieces.append(flat[base : base + valid_elems])
        return np.concatenate(pieces) if pieces else flat[:0]

    return np.asarray(arr)[:vr, :vc]
