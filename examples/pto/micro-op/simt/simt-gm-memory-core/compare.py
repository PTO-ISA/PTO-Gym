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


def main():
    golden = np.fromfile("golden_v1.bin", dtype=np.int32)
    out = np.fromfile("v1.bin", dtype=np.int32)
    if golden.shape != out.shape or not np.array_equal(golden, out):
        idxs = np.nonzero(golden != out)[0]
        idx = int(idxs[0]) if idxs.size else 0
        print(
            f"[ERROR] mismatch at idx={idx}, golden={int(golden[idx])}, out={int(out[idx])}"
        )
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
