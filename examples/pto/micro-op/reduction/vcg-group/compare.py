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


def compare_bin(golden_path, output_path, eps):
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False

    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        idx = int(np.argmax(diff))
        print(
            f"[ERROR] Mismatch: {golden_path} vs {output_path}, "
            f"idx={idx}, golden={golden[idx]}, output={output[idx]}, max_diff={diff[idx]}"
        )
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    checks = [
        ("golden_add.bin", "out_add.bin", "vcgadd"),
        ("golden_max.bin", "out_max.bin", "vcgmax"),
        ("golden_min.bin", "out_min.bin", "vcgmin"),
    ]
    failed = []
    for golden, output, label in checks:
        if not compare_bin(golden, output, 1e-4):
            failed.append(label)
            print(f"[ERROR] compare failed: {label}")
    if failed:
        if strict:
            print(f"[ERROR] {len(failed)} check(s) failed: {', '.join(failed)}")
            sys.exit(2)
        print(f"[WARN] {len(failed)} check(s) failed (non-gating): {', '.join(failed)}")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
