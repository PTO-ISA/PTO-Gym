#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

import numpy as np


EXPECTED_ELEMS = 64 + 128


def compare_one(golden_path: Path, actual_path: Path, label: str) -> bool:
    if not golden_path.exists() or not actual_path.exists():
        print(f"[ERROR] {label}: missing {golden_path} or {actual_path}")
        return False
    golden = np.fromfile(golden_path, dtype=np.uint32)
    actual = np.fromfile(actual_path, dtype=np.uint32)
    if golden.size != EXPECTED_ELEMS or actual.size != EXPECTED_ELEMS:
        print(
            f"[ERROR] {label}: expected {EXPECTED_ELEMS} elements, "
            f"got golden={golden.size}, actual={actual.size}"
        )
        return False
    mismatches = np.flatnonzero(golden != actual)
    if mismatches.size == 0:
        return True
    print(f"[ERROR] {label}: {mismatches.size} mismatched lanes")
    for index in mismatches[:16]:
        case = "signed64" if index < 64 else "unsigned128"
        lane = int(index) if index < 64 else int(index) - 64
        print(
            f"  case={case} lane={lane}: "
            f"expected=0x{int(golden[index]):08x}, actual=0x{int(actual[index]):08x}"
        )
    return False


def main() -> None:
    ok_low = compare_one(Path("golden_low.bin"), Path("low.bin"), "low")
    ok_high = compare_one(Path("golden_high.bin"), Path("high.bin"), "high")
    if not (ok_low and ok_high):
        sys.exit(2)
    print("[INFO] compare passed: signed sparse-mask and unsigned multi-register VMULL")


if __name__ == "__main__":
    main()
