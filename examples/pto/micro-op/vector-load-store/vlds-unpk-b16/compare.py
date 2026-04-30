#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vlds-unpk-b16
# family: vector-load-store
# target_ops: pto.vlds
# scenarios: core-f16, full-mask, aligned, dist-unpk-b16

import os
import sys
import numpy as np


def compare_bin(golden_path, output_path):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    golden = np.fromfile(golden_path, dtype=np.uint16)
    output = np.fromfile(output_path, dtype=np.uint16)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        return False
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch: {golden_path} vs {output_path}, first diff at idx={idx} "
            f"(golden=0x{int(golden[idx]):04x}, out=0x{int(output[idx]):04x})"
        )
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v2.bin", "v2.bin")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
