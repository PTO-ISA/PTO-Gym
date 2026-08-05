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


def main() -> None:
    golden_path = "golden_output.bin"
    output_path = "output.bin"
    strict = os.getenv("COMPARE_STRICT", "1") != "0"

    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        print("[ERROR] missing golden_output.bin or output.bin")
        sys.exit(2 if strict else 0)

    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    ok = golden.shape == output.shape and np.allclose(
        golden, output, atol=0.0001, rtol=0.0001, equal_nan=True
    )
    if not ok:
        if golden.shape != output.shape:
            print(f"[ERROR] shape mismatch: {golden.shape} vs {output.shape}")
        elif golden.size:
            diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
            idx = int(np.argmax(diff))
            print(
                f"[ERROR] mismatch at idx={idx}: golden={golden[idx]} "
                f"output={output[idx]} diff={diff[idx]}"
            )
        if strict:
            sys.exit(2)
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
