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


def check_f32() -> bool:
    golden = np.fromfile("golden_v2.bin", dtype=np.float32)
    output = np.fromfile("v2.bin", dtype=np.float32)
    if golden.shape == output.shape and np.allclose(golden, output, atol=1e-5, rtol=1e-5):
        return True
    diff = np.nonzero(~np.isclose(golden, output, atol=1e-5, rtol=1e-5))[0]
    idx = int(diff[0]) if diff.size else -1
    print(
        f"[ERROR] compare failed v2 idx={idx} "
        f"golden={golden[idx] if idx >= 0 else 'n/a'} "
        f"output={output[idx] if idx >= 0 else 'n/a'}"
    )
    return False


def check_f16() -> bool:
    golden = np.fromfile("golden_v3.bin", dtype=np.float16)
    output = np.fromfile("v3.bin", dtype=np.float16)
    if golden.shape == output.shape and np.array_equal(golden, output):
        return True
    diff = np.nonzero(golden.view(np.uint16) != output.view(np.uint16))[0]
    idx = int(diff[0]) if diff.size else -1
    print(
        f"[ERROR] compare failed v3 idx={idx} "
        f"golden={golden[idx] if idx >= 0 else 'n/a'} "
        f"output={output[idx] if idx >= 0 else 'n/a'}"
    )
    return False


def main() -> None:
    if not check_f32() or not check_f16():
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
