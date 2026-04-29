#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Compare golden and output for trowexpanddiv ST test cases."""

import os
import sys
import numpy as np

from cases import CASES
from st_common import result_cmp, style_fail, style_pass

# Inline validation for multi-input format (trowexpanddiv uses src0/src1/dst)
REQUIRED_KEYS = {"name", "dtype", "src0_shape", "src0_valid_shape", "src1_shape",
                 "src1_valid_shape", "dst_shape", "dst_valid_shape"}
for i, case in enumerate(CASES):
    missing = REQUIRED_KEYS - case.keys()
    if missing:
        raise ValueError(f"cases[{i}] ({case.get('name', '?')}) missing keys: {missing}")


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        dst_shape = case["dst_shape"]
        dst_valid_shape = case["dst_valid_shape"]
        dtype = case["dtype"]

        vr, vc = dst_valid_shape

        golden = np.fromfile(os.path.join(case_dir, "golden.bin"), dtype=dtype).reshape(dst_shape)
        output = np.fromfile(os.path.join(case_dir, "output.bin"), dtype=dtype).reshape(dst_shape)

        ok = result_cmp(golden[:vr, :vc], output[:vr, :vc], case["eps"])
        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()