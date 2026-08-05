#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import numpy as np


def check(output_name: str, golden_name: str) -> None:
    golden = np.fromfile(golden_name, dtype=np.float32)
    output = np.fromfile(output_name, dtype=np.float32)
    if golden.shape == output.shape and np.allclose(golden, output, atol=1e-4, rtol=1e-4):
        return

    if golden.shape != output.shape:
        print(f"[ERROR] compare failed {output_name}: shape golden={golden.shape} output={output.shape}")
        sys.exit(2)
    diff = np.nonzero(~np.isclose(golden, output, atol=1e-4, rtol=1e-4))[0]
    idx = int(diff[0]) if diff.size else -1
    print(
        f"[ERROR] compare failed {output_name} idx={idx} "
        f"golden={golden[idx] if idx >= 0 else 'n/a'} "
        f"output={output[idx] if idx >= 0 else 'n/a'}"
    )
    sys.exit(2)


def main() -> None:
    check("v4.bin", "golden_v4.bin")
    check("v5.bin", "golden_v5.bin")
    check("v6.bin", "golden_v6.bin")
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
