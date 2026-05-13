#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np

from cases import CASES
from st_common import result_cmp, style_fail, style_pass, validate_cases


def main():
    validate_cases(CASES)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        shape = case["shape"]
        vr, vc = case["valid_shape"]
        dtype = case["dtype"]
        out_dtype = case["out_dtype"]
        elem_size = np.dtype(dtype).itemsize
        lanes = 256 // elem_size
        
        # Calculate expected output size (same as gen_data.py)
        total_elm = vr * vc
        if elem_size == 4:  # 32B
            bytes_per_iter = 16
            repeat_times = (total_elm + lanes - 1) // lanes + 1
            total_iters = repeat_times // 2
            expected_bytes = total_iters * bytes_per_iter
        elif elem_size == 2:  # 16B
            bytes_per_iter = 16
            iters_per_row = (vc + lanes - 1) // lanes
            expected_bytes = vr * iters_per_row * bytes_per_iter
        else:  # 8B
            bytes_per_iter = 32
            iters_per_row = (vc + lanes - 1) // lanes
            expected_bytes = vr * iters_per_row * bytes_per_iter
        
        # Read golden (already correct size from gen_data.py)
        golden = np.fromfile(os.path.join(case_dir, "golden.bin"), dtype=np.uint8)
        
        # Read output and truncate/zero-pad to expected size
        output = np.fromfile(os.path.join(case_dir, "output.bin"), dtype=np.uint8)
        if len(output) > expected_bytes:
            output = output[:expected_bytes]
        elif len(output) < expected_bytes:
            output = np.pad(output, (0, expected_bytes - len(output)), mode='constant')

        # Compare byte-by-byte
        ok = np.array_equal(golden, output)
        if not ok:
            # Find first mismatch for debugging
            diff_mask = golden != output
            diff_indices = np.where(diff_mask)[0]
            if len(diff_indices) > 0:
                diff_idx = diff_indices[0]
                max_diff = int(np.max(np.abs(golden.astype(int) - output.astype(int))))
                print(style_fail(f"[ERROR] Mismatch: max diff={max_diff} at byte idx={diff_idx} "
                                 f"(golden=0x{golden[diff_idx]:02x}, output=0x{output[diff_idx]:02x})"))
            else:
                print(style_fail(f"[ERROR] Mismatch: shapes differ golden={golden.shape} output={output.shape}"))

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
