#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/compare-select/vselr-f16
# family: compare-select
# target_ops: pto.vselr
# scenarios: core-f16, full-mask, explicit-lane-index

import os
import sys

import numpy as np


def compare_tensor(golden_path: str, output_path: str) -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=np.float16)
    output = np.fromfile(output_path, dtype=np.float16)
    if golden.shape != output.shape:
        return False
    if not np.allclose(golden, output, rtol=0.0, atol=0.0, equal_nan=True):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] Mismatch: idx={idx} golden={golden[idx]} out={output[idx]}")
        return False
    return True


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_tensor("golden_v3.bin", "v3.bin")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
