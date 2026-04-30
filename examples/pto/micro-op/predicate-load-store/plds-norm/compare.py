#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/predicate-load-store/plds-norm
# family: predicate-load-store
# target_ops: pto.plds
# scenarios: packed-load, dynamic-offset, representative-logical-elements

import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=np.uint8)
    output = np.fromfile("v2.bin", dtype=np.uint8)
    if golden.size < 256 or output.size < 256:
        print(
            f"[ERROR] Packed buffer too small: golden={golden.size} out={output.size}"
        )
        raise SystemExit(2)
    if not np.array_equal(golden[:256], output[:256]):
        diff = np.nonzero(golden[:256] != output[:256])[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (plds NORM -> vsel): idx={idx} "
            f"golden={int(golden[idx])} out={int(output[idx])}"
        )
        raise SystemExit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
