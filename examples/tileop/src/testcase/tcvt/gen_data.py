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
import ml_dtypes

from cases import CASES
from compare import normalize_dtype
from st_common import save_case_data, setup_case_rng, validate_cases


def is_sub_float(dtype):
    return np.issubdtype(dtype, np.floating) or dtype == ml_dtypes.bfloat16


def is_sub_int(dtype):
    return np.issubdtype(dtype, np.integer)


def _make_input_inner(src_dtype, shape):
    total = int(np.prod(shape))
    float_types = (np.float32, np.float16, ml_dtypes.bfloat16)
    int8_like_types = (np.int8, )

    # Generate input data
    if src_dtype in float_types:
        return (np.random.random([total]) * 200 - 100)
    elif src_dtype in int8_like_types:
        return np.random.randint(-128, 128, [total])
    elif src_dtype == np.uint8:
        return np.random.randint(0, 256, [total])
    elif src_dtype == np.int16:
        return np.random.randint(-1000, 1000, [total])
    elif src_dtype == np.uint16:
        return np.random.randint(0, 10000, [total])
    elif src_dtype in (np.int32, np.int64):
        return np.random.randint(-10000, 10000, [total])
    elif src_dtype == np.uint32:
        return np.random.randint(0, 10000, [total])
    else:
        return np.random.randint(-10000, 10000, [total])


def make_input(src_dtype, shape):
    return _make_input_inner(src_dtype, shape).astype(normalize_dtype(src_dtype)).reshape(shape)


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


def convert(values: np.ndarray, src_dtype, dst_dtype, round_mode=None):
    is_float_src = is_sub_float(src_dtype)
    is_int_dst = is_sub_int(dst_dtype)
    is_f32_to_f32 = src_dtype == np.float32 and dst_dtype == np.float32
    needs_rounding = is_float_src and (is_int_dst or is_f32_to_f32)

    if needs_rounding:
        values = apply_round_mode(values, round_mode or "RINT")
    
    if is_int_dst:
        # Determine if this conversion has default saturation OFF (truncation) or ON (clamping)
        if default_saturation_off(src_dtype, dst_dtype):
            # OFF (truncation): bit extraction - wrap around using modulo
            return truncate_to_int(values, dst_dtype)
        else:
            # Saturation ON: clamp to range (widen to int64/float64 to preserve sign)
            return clamp_to_range_int(values, dst_dtype)
    elif is_sub_float(dst_dtype):
        return clamp_to_range_float(values, dst_dtype)
    else:
        return values.astype(dst_dtype)


def truncate_to_int(values: np.ndarray, dst_dtype):
    golden_list = []
    for val in values.flat:
        int_val = 0 if np.isnan(val) or np.isinf(val) else int(np.int64(val))

        if dst_dtype == np.int8:
            byte_val = int_val & 0xFF
            truncated_val = byte_val if byte_val < 128 else byte_val - 256
        elif dst_dtype == np.uint8:
            truncated_val = int_val & 0xFF
        elif dst_dtype == np.int16:
            word_val = int_val & 0xFFFF
            truncated_val = word_val if word_val < 32768 else word_val - 65536
        elif dst_dtype == np.int32:
            dword_val = int_val & 0xFFFFFFFF
            truncated_val = dword_val if dword_val < 2147483648 else dword_val - 4294967296
        else:
            truncated_val = int_val
        golden_list.append(truncated_val)
    return np.array(golden_list, dtype=dst_dtype).reshape(values.shape)


def clamp_to_range_int(values: np.ndarray, dst_dtype):
    info = ml_dtypes.iinfo(dst_dtype)
    is_int_type = is_sub_int(values.dtype)
    temp_dtype = np.int64 if is_int_type else np.float64
    widened = values.astype(temp_dtype, copy=False)
    return np.clip(widened, info.min, info.max).astype(dst_dtype)


def clamp_to_range_float(values: np.ndarray, dst_dtype):
    info = ml_dtypes.finfo(dst_dtype)
    return np.clip(values, info.min, info.max).astype(dst_dtype)


def apply_valid_shape(values: np.ndarray, valid_shape, dst_dtype):
    vr, vc = valid_shape
    masked = np.zeros_like(values, dtype=dst_dtype)
    masked[:vr, :vc] = values[:vr, :vc]
    return masked

def generate_golden(case):
    src_dtype = case["src_dtype"]
    dst_dtype = case["dst_dtype"]
    src_dtype_norm = normalize_dtype(src_dtype)
    dst_dtype_norm = normalize_dtype(dst_dtype)
    shape = case["shape"]
    round_mode = case.get("round_mode")

    input_arr = make_input(src_dtype, shape)
    converted = convert(input_arr, src_dtype_norm, dst_dtype_norm, round_mode)
    golden = apply_valid_shape(converted, case["valid_shape"], dst_dtype_norm)

    return input_arr, golden


if __name__ == "__main__":
    np.random.seed(19)

    validate_cases(CASES)

    for case in CASES:
        setup_case_rng(case)
        input_arr, golden = generate_golden(case)

        save_case_data(case["name"], {"input": input_arr, "golden": golden})
        src_dtype = case["src_dtype"]
        dst_dtype = case["dst_dtype"]
        src_name = src_dtype.__name__ if isinstance(src_dtype, type) else src_dtype
        dst_name = dst_dtype.__name__ if isinstance(dst_dtype, type) else dst_dtype
        print(
            f"[INFO] gen_data: {case['name']} shape={case['shape']} "
            f"src_dtype={src_name} dst_dtype={dst_name} "
            f"round_mode={case.get('round_mode')}"
        )
