#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Shared utilities for TileLang ST test cases.

Provides:
  - Data helpers:  setup_case_rng(), save_case_data()
  - Compare:       result_cmp()
  - Styling:       supports_color(), style_pass(), style_fail()
"""

import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Case helpers
# ---------------------------------------------------------------------------

REQUIRED_CASE_KEYS = {"name", "dtype", "shape", "valid_shape", "eps"}


def _to_shape_tuple(shape):
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"shape must be tuple/list, got {type(shape).__name__}: {shape!r}")
    if not shape:
        raise ValueError("shape must not be empty")
    dims = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"shape dimensions must be > 0, got {dims}")
    return dims


def _validate_shape_pair(shape, valid_shape, label):
    shape = _to_shape_tuple(shape)
    valid_shape = _to_shape_tuple(valid_shape)
    if len(shape) != len(valid_shape):
        raise ValueError(f"{label}: shape rank mismatch: {shape} vs {valid_shape}")
    if any(valid_dim > dim for dim, valid_dim in zip(shape, valid_shape)):
        raise ValueError(f"{label}: valid shape {valid_shape} exceeds shape {shape}")
    return shape, valid_shape


def validate_cases(cases):
    """Check that every case has all required keys."""
    for i, case in enumerate(cases):
        missing = REQUIRED_CASE_KEYS - case.keys()
        if missing:
            raise ValueError(f"cases[{i}] ({case.get('name', '?')}) missing keys: {missing}")
        _validate_shape_pair(case["shape"], case["valid_shape"], "shape")
        has_dst_shape = "dst_shape" in case
        has_dst_valid_shape = "dst_valid_shape" in case
        if has_dst_shape != has_dst_valid_shape:
            raise ValueError(
                f"cases[{i}] ({case.get('name', '?')}) must define both dst_shape and dst_valid_shape"
            )
        if has_dst_shape:
            _validate_shape_pair(case["dst_shape"], case["dst_valid_shape"], "dst")


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def setup_case_rng(case):
    """Set a per-case deterministic random seed.

    Using hash(name) ensures that adding/reordering cases does not change
    the random data of existing cases.
    """
    np.random.seed(hash(case["name"]) & 0xFFFFFFFF)


def save_case_data(case_name, data_dict):
    """Create case directory and write {name}.bin for each entry in data_dict.

    Args:
        case_name: subdirectory name (e.g. "f32_16x64").
        data_dict: mapping from file stem to numpy array,
                   e.g. {"input1": arr1, "input2": arr2, "golden": golden}.
    """
    os.makedirs(case_name, exist_ok=True)
    for name, arr in data_dict.items():
        arr.tofile(os.path.join(case_name, f"{name}.bin"))


# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------

ANSI_RESET = "\033[0m"
ANSI_BOLD_GREEN = "\033[1;32m"
ANSI_BOLD_RED = "\033[1;31m"


def supports_color():
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "", "dumb")


def style_pass(text):
    if not supports_color():
        return text
    return f"{ANSI_BOLD_GREEN}{text}{ANSI_RESET}"


def style_fail(text):
    if not supports_color():
        return text
    return f"{ANSI_BOLD_RED}{text}{ANSI_RESET}"


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def result_cmp(golden, output, eps):
    """Compare already prepared golden/output arrays.

    The caller is responsible for loading, reshaping and slicing data.
    """
    g = np.asarray(golden).astype(np.float64, copy=False)
    o = np.asarray(output).astype(np.float64, copy=False)

    if g.shape != o.shape:
        print(style_fail(f"[ERROR] Shape mismatch: golden {g.shape} vs output {o.shape}"))
        return False
    if not np.allclose(g, o, atol=eps, rtol=eps, equal_nan=True):
        abs_diff = np.abs(g - o)
        idx = int(np.argmax(abs_diff))
        print(style_fail(f"[ERROR] Mismatch: max diff={float(abs_diff.flat[idx])} "
                         f"at flat idx={idx} "
                         f"(golden={g.flat[idx]}, output={o.flat[idx]})"))
        return False
    return True
