#!/usr/bin/python3
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


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    golden = np.fromfile("golden_v3.bin", dtype=np.uint8)
    output = np.fromfile("v3.bin", dtype=np.uint8)
    ok = golden.size >= 32 and output.size >= 32 and np.array_equal(golden[:32], output[:32])
    if not ok:
        if golden.size and output.size:
            diff = np.nonzero(golden[:32] != output[:32])[0]
            idx = int(diff[0]) if diff.size else 0
            print(f"[ERROR] Mismatch: idx={idx} golden={int(golden[idx])} out={int(output[idx])}")
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
