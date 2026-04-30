#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/rearrangement/vusqz
# family: rearrangement
# target_ops: pto.vusqz
# scenarios: predicate-driven-rearrangement, prefix-count

import sys
import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v3.bin", dtype=np.int32)
    output = np.fromfile("v3.bin", dtype=np.int32)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        sys.exit(2)
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch at idx={idx}: golden={int(golden[idx])} out={int(output[idx])}"
        )
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
