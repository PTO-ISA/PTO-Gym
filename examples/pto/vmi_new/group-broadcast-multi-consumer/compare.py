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
    golden_sum = np.fromfile("golden_v2.bin", dtype=np.float32)
    output_sum = np.fromfile("v2.bin", dtype=np.float32)
    if golden_sum.shape != output_sum.shape or not np.allclose(golden_sum, output_sum, atol=1e-4, rtol=1e-4):
        diff = np.nonzero(~np.isclose(golden_sum, output_sum, atol=1e-4, rtol=1e-4))[0]
        idx = int(diff[0]) if diff.size else -1
        print(
            f"[ERROR] compare failed v2 idx={idx} "
            f"golden={golden_sum[idx] if idx >= 0 else 'n/a'} "
            f"output={output_sum[idx] if idx >= 0 else 'n/a'}"
        )
        sys.exit(2)

    golden_dense = np.fromfile("golden_v3.bin", dtype=np.float16)
    output_dense = np.fromfile("v3.bin", dtype=np.float16)
    if golden_dense.shape != output_dense.shape or not np.array_equal(golden_dense, output_dense):
        diff = np.nonzero(golden_dense.view(np.uint16) != output_dense.view(np.uint16))[0]
        idx = int(diff[0]) if diff.size else -1
        print(
            f"[ERROR] compare failed v3 idx={idx} "
            f"golden={golden_dense[idx] if idx >= 0 else 'n/a'} "
            f"output={output_dense[idx] if idx >= 0 else 'n/a'}"
        )
        sys.exit(2)

    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
