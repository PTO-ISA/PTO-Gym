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


def main() -> None:
    golden_scale = np.fromfile("golden_v2.bin", dtype=np.float32)
    scale = np.fromfile("v2.bin", dtype=np.float32)
    golden_out = np.fromfile("golden_v3.bin", dtype=np.uint8)
    out = np.fromfile("v3.bin", dtype=np.uint8)

    if golden_scale.shape != scale.shape or not np.allclose(
        golden_scale, scale, rtol=1.0e-6, atol=1.0e-6
    ):
        if golden_scale.shape != scale.shape:
            idx = -1
        else:
            diff = np.nonzero(~np.isclose(golden_scale, scale, rtol=1.0e-6, atol=1.0e-6))[0]
            idx = int(diff[0]) if diff.size else -1
        print(
            f"[ERROR] scale compare failed idx={idx} "
            f"golden={golden_scale[idx] if idx >= 0 else 'n/a'} "
            f"output={scale[idx] if idx >= 0 else 'n/a'}"
        )
        sys.exit(2)

    if golden_out.shape != out.shape or not np.array_equal(golden_out, out):
        diff = np.nonzero(golden_out != out)[0] if golden_out.shape == out.shape else []
        idx = int(diff[0]) if len(diff) else -1
        print(
            f"[ERROR] int8 compare failed idx={idx} "
            f"golden={int(golden_out[idx]) if idx >= 0 else 'n/a'} "
            f"output={int(out[idx]) if idx >= 0 else 'n/a'}"
        )
        sys.exit(2)

    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
