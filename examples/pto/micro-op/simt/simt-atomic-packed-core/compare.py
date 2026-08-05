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


def compare_one(name: str) -> bool:
    golden = np.fromfile(f"golden_{name}.bin", dtype=np.uint32)
    out = np.fromfile(f"{name}.bin", dtype=np.uint32)
    ok = golden.shape == out.shape and np.array_equal(golden, out)
    if not ok:
        idxs = np.nonzero(golden != out)[0]
        idx = int(idxs[0]) if idxs.size else 0
        print(
            f"[ERROR] {name} mismatch at idx={idx}, "
            f"golden=0x{int(golden[idx]):08x}, out=0x{int(out[idx]):08x}"
        )
    return ok


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_one("v1") and compare_one("v2")
    if not ok and strict:
        sys.exit(2)
    print("[INFO] compare passed" if ok else "[WARN] compare failed (non-gating)")


if __name__ == "__main__":
    main()
