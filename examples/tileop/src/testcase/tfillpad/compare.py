#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Compare output against golden for tfillpad test cases.

For tfillpad:
  - Input: full tile shape (rows x cols)
  - Output: only valid region (valid_rows x valid_cols)
  - Golden: valid region only
"""

import os
import sys
import numpy as np

from cases import CASES


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        dtype = case["dtype"]
        valid_shape = case["valid_shape"]
        eps = case["eps"]

        # Load golden and output (both stored with valid_shape)
        golden = np.fromfile(os.path.join(case_dir, "golden.bin"), dtype=dtype).reshape(valid_shape)
        output = np.fromfile(os.path.join(case_dir, "output.bin"), dtype=dtype).reshape(valid_shape)

        # For integer types, eps=0 means exact match
        # For float types, use np.allclose with eps
        if eps == 0:
            # Integer comparison - exact match
            if not np.array_equal(golden, output):
                diff = golden - output
                idx = int(np.argmax(np.abs(diff)))
                print(f"[ERROR] {case['name']}: Mismatch at idx={idx} (golden={golden.flat[idx]}, output={output.flat[idx]})")
                all_passed = False
            else:
                print(f"[INFO] {case['name']}: compare passed")
        else:
            # Float comparison - use allclose
            # Convert to float64 for comparison (fp16 precision issues)
            g = golden.astype(np.float64, copy=False)
            o = output.astype(np.float64, copy=False)

            if g.shape != o.shape:
                print(f"[ERROR] {case['name']}: Shape mismatch: golden {g.shape} vs output {o.shape}")
                all_passed = False
                continue

            if not np.allclose(g, o, atol=eps, rtol=eps, equal_nan=True):
                abs_diff = np.abs(g - o)
                idx = int(np.argmax(abs_diff))
                print(f"[ERROR] {case['name']}: Mismatch: max diff={float(abs_diff.flat[idx])} "
                      f"at idx={idx} (golden={g.flat[idx]}, output={o.flat[idx]})")
                all_passed = False
            else:
                print(f"[INFO] {case['name']}: compare passed")

    if not all_passed:
        sys.exit(2)
    print("[INFO] all cases passed")


if __name__ == "__main__":
    main()