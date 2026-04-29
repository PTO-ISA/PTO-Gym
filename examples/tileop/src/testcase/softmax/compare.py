#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import numpy as np
from cases import CASES
from st_common import result_cmp, style_fail, style_pass, validate_cases


def load_array(path, dtype, shape):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=dtype).reshape(shape)


def compare_case(case):
    case_dir = case["name"]
    rows = int(case["rows"])
    cols = int(case["cols"])
    seq = int(case["seq"])
    dtype = case["dtype"]
    eps = case["eps"]

    try:
        golden_v4 = load_array(os.path.join(case_dir, "golden_v4.bin"), dtype, (rows,))
        output_v4 = load_array(os.path.join(case_dir, "v4.bin"), dtype, (rows,))
        golden_v5 = load_array(os.path.join(case_dir, "golden_v5.bin"), dtype, (rows,))
        output_v5 = load_array(os.path.join(case_dir, "v5.bin"), dtype, (rows,))
        golden_v6 = load_array(os.path.join(case_dir, "golden_v6.bin"), dtype, (rows,))
        output_v6 = load_array(os.path.join(case_dir, "v6.bin"), dtype, (rows,))
        golden_v7 = load_array(
            os.path.join(case_dir, "golden_v7.bin"), dtype, (rows, cols)
        )
        output_v7 = load_array(os.path.join(case_dir, "v7.bin"), dtype, (rows, cols))
    except FileNotFoundError as exc:
        print(style_fail(f"[ERROR] {case['name']}: missing file {exc}"))
        return False

    ok = True
    ok = result_cmp(golden_v4, output_v4, eps) and ok
    ok = result_cmp(golden_v5, output_v5, eps) and ok
    ok = result_cmp(golden_v6, output_v6, eps) and ok
    ok = result_cmp(golden_v7[:, :seq], output_v7[:, :seq], eps) and ok
    return ok


def main():
    validate_cases(CASES)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    matched_case = case_filter is None
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        matched_case = True
        ok = compare_case(case)
        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not matched_case:
        print(style_fail(f"[ERROR] unknown case filter: {case_filter}"))
        sys.exit(2)
    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()
