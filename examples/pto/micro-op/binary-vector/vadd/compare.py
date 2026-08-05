#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Merged vadd compare: checks all 9 variants.

import os
import sys
import numpy as np


def compare_bin(golden_path, output_path, dtype, eps, count=-1):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    kw = {} if count < 0 else {"count": count}
    golden = np.fromfile(golden_path, dtype=dtype, **kw)
    output = np.fromfile(output_path, dtype=dtype, **kw)
    return golden.shape == output.shape and np.allclose(
        golden, output, atol=eps, rtol=eps, equal_nan=True
    )


def compare_bin_exact(golden_path, output_path, dtype):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    return golden.shape == output.shape and np.array_equal(golden, output)


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"

    # (golden, output, dtype, eps, count, label)
    checks = [
        ("golden_v3.bin",      "v3.bin",      np.float32, 1e-4,  -1,   "f32"),
        ("golden_v3_f16.bin",  "v3_f16.bin",  np.float16, 5e-3,  1024, "f16"),
        ("golden_v3_bf16.bin", "v3_bf16.bin", np.uint16,  0,     1024, "bf16"),
        ("golden_v3_x.bin",    "v3_x.bin",    np.float32, 1e-4,  -1,   "f32-exceptional"),
        ("golden_v3_i16s.bin", "v3_i16s.bin", np.int16,   0,     1024, "i16-signed"),
        ("golden_v3_tail.bin", "v3_tail.bin", np.float32, 1e-4,  1000, "tail"),
    ]
    # Overflow variants need exact match (wrapping arithmetic)
    checks_exact = [
        ("golden_v3_i16s_ov.bin", "v3_i16s_ov.bin", np.int16,   "i16-signed-overflow"),
        ("golden_v3_i16u.bin",    "v3_i16u.bin",    np.uint16,  "i16-unsigned"),
        ("golden_v3_i16u_ov.bin", "v3_i16u_ov.bin", np.uint16,  "i16-unsigned-overflow"),
    ]

    failed = []
    for golden, output, dtype, eps, count, label in checks:
        ok = compare_bin(golden, output, dtype, eps, count)
        if not ok:
            failed.append(label)
            print(f"[ERROR] compare failed: {label}")

    for golden, output, dtype, label in checks_exact:
        ok = compare_bin_exact(golden, output, dtype)
        if not ok:
            failed.append(label)
            print(f"[ERROR] compare failed (exact): {label}")

    if failed:
        if strict:
            print(f"[ERROR] {len(failed)} variant(s) failed: {', '.join(failed)}")
            sys.exit(2)
        print(f"[WARN] {len(failed)} variant(s) failed (non-gating): {', '.join(failed)}")
        return
    print("[INFO] compare passed (all 9 variants)")


if __name__ == "__main__":
    main()
