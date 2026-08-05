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

CHECK_ELEMS = 128


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    golden = np.fromfile("golden_v1.bin", dtype=np.int32)
    out = np.fromfile("v1.bin", dtype=np.int32)
    golden_prefix = golden[:CHECK_ELEMS]
    out_prefix = out[:CHECK_ELEMS]
    shape_ok = golden_prefix.shape == (CHECK_ELEMS,) and out_prefix.shape == (
        CHECK_ELEMS,
    )
    ok = shape_ok and np.array_equal(golden_prefix, out_prefix)
    if not ok:
        if not shape_ok:
            print(
                f"[ERROR] expected at least {CHECK_ELEMS} elements, "
                f"golden={golden.size}, out={out.size}"
            )
            if strict:
                sys.exit(2)
            print(f"[WARN] compare failed for first {CHECK_ELEMS} elements (non-gating)")
            return
        idxs = np.nonzero(golden_prefix != out_prefix)[0]
        idx = int(idxs[0]) if idxs.size else 0
        print(
            f"[ERROR] mismatch at idx={idx}, golden={int(golden_prefix[idx])}, out={int(out_prefix[idx])}"
        )
        if strict:
            sys.exit(2)
    print(
        f"[INFO] compare passed for first {CHECK_ELEMS} elements"
        if ok
        else f"[WARN] compare failed for first {CHECK_ELEMS} elements (non-gating)"
    )


if __name__ == "__main__":
    main()
