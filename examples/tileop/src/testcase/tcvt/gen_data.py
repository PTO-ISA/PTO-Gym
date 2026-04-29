#!/usr/bin/python3
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
from st_common import save_case_data, setup_case_rng, validate_cases


def make_f32_input(shape):
    total = int(np.prod(shape))
    base = (np.arange(total, dtype=np.float32) % 17) - 8.0
    frac_table = np.array([0.2, 0.5, 0.8, -0.2, -0.5, -0.8], dtype=np.float32)
    frac = frac_table[np.arange(total) % frac_table.size]
    return (base + frac).reshape(shape)


def make_i32_input(shape):
    total = int(np.prod(shape))
    return (((np.arange(total, dtype=np.int32) * 37) % 257) - 128).reshape(shape)


def round_half_away_from_zero(values):
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def default_saturation_off(src_dtype, dst_dtype):
    """Mirror the current A5 default saturation policy for supported pairs."""
    return (
        (src_dtype is np.float16 and dst_dtype is np.uint8)
        or (src_dtype is np.float16 and dst_dtype is np.int8)
        or (src_dtype is np.float32 and dst_dtype is np.int16)
        or (src_dtype is np.float16 and dst_dtype is np.int16)
        or (src_dtype is np.int64 and dst_dtype is np.int32)
        or (src_dtype is np.int32 and dst_dtype is np.int16)
    )


def apply_round_mode(values, round_mode):
    rounding_funcs = {
        "RINT": np.rint,
        "ROUND": round_half_away_from_zero,
        "FLOOR": np.floor,
        "CEIL": np.ceil,
        "TRUNC": np.trunc,
    }
    return rounding_funcs.get(round_mode, np.rint)(values)


def convert_with_default_saturation(values, src_dtype, dst_dtype):
    if np.issubdtype(dst_dtype, np.integer):
        if default_saturation_off(src_dtype, dst_dtype):
            # For currently supported ST cases this branch is not taken, but keep
            # the structure aligned with pto-isa's A5 tcvt golden generator.
            if dst_dtype is np.int32:
                widened = values.astype(np.int64, copy=False)
                wrapped = np.where(widened < 0, (widened + (1 << 32)) & 0xFFFFFFFF, widened & 0xFFFFFFFF)
                signed = np.where(wrapped < (1 << 31), wrapped, wrapped - (1 << 32))
                return signed.astype(np.int32, copy=False)
            return values.astype(dst_dtype, copy=False)
        info = np.iinfo(dst_dtype)
        widened = values.astype(np.float64, copy=False)
        return np.clip(widened, info.min, info.max).astype(dst_dtype)

    if np.issubdtype(dst_dtype, np.floating):
        info = np.finfo(dst_dtype)
        return np.clip(values.astype(np.float64, copy=False), info.min, info.max).astype(dst_dtype)

    return values.astype(dst_dtype, copy=False)


def generate_golden(case):
    src_dtype = case["src_dtype"]
    dst_dtype = case["dst_dtype"]
    shape = case["shape"]
    vr, vc = case["valid_shape"]

    if src_dtype is np.float32 and dst_dtype is np.int32:
        input_arr = make_f32_input(shape).astype(src_dtype)
        rounded = apply_round_mode(input_arr[:vr, :vc], case["round_mode"])
        converted = convert_with_default_saturation(rounded, src_dtype, dst_dtype)
    elif src_dtype is np.float32 and dst_dtype is np.float16:
        input_arr = make_f32_input(shape).astype(src_dtype)
        converted = convert_with_default_saturation(input_arr[:vr, :vc], src_dtype, dst_dtype)
    elif src_dtype is np.int32:
        input_arr = make_i32_input(shape).astype(src_dtype)
        converted = convert_with_default_saturation(input_arr[:vr, :vc], src_dtype, dst_dtype)
    elif src_dtype is np.float16:
        input_arr = make_f32_input(shape).astype(src_dtype)
        converted = convert_with_default_saturation(input_arr[:vr, :vc], src_dtype, dst_dtype)
    else:
        raise TypeError(f"unsupported tcvt ST source dtype: {src_dtype}")

    golden = np.zeros(shape, dtype=dst_dtype)
    golden[:vr, :vc] = converted
    return input_arr, golden


validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)
    input_arr, golden = generate_golden(case)

    save_case_data(case["name"], {"input": input_arr, "golden": golden})
    print(
        f"[INFO] gen_data: {case['name']} shape={case['shape']} "
        f"src_dtype={case['src_dtype'].__name__} dst_dtype={case['dst_dtype'].__name__} "
        f"round_mode={case['round_mode']}"
    )
