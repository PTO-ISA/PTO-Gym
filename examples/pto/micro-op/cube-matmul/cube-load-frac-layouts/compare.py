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


CASES = (
    ("golden_nd2nz_case1.bin", "out_nd2nz_case1.bin"),
)


def compare_one(golden_path: str, output_path: str) -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        print(f"[ERROR] missing file: {golden_path} or {output_path}")
        return False
    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    if golden.shape != output.shape:
        print(f"[ERROR] shape mismatch for {output_path}: {golden.shape} vs {output.shape}")
        return False
    if np.allclose(golden, output, atol=1e-3, rtol=1e-3):
        return True
    diff = np.nonzero(~np.isclose(golden, output, atol=1e-3, rtol=1e-3))[0]
    idx = int(diff[0]) if diff.size else 0
    print(f"[ERROR] {output_path} mismatch at idx={idx}: golden={golden[idx]}, out={output[idx]}")
    return False


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = all(compare_one(golden_path, output_path) for golden_path, output_path in CASES)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
