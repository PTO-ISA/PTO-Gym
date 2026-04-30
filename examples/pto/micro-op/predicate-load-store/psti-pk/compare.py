#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/predicate-load-store/psti-pk
# family: predicate-load-store
# target_ops: pto.psti
# scenarios: packed-store, immediate-offset, representative-logical-elements

import numpy as np


EXPECTED_WORDS = 8
PK_STORAGE_BYTES = 16


def main() -> None:
    golden = np.fromfile("golden_v1.bin", dtype=np.uint8)
    output = np.fromfile("v1.bin", dtype=np.uint8)
    expected_bytes = EXPECTED_WORDS * 4
    if golden.size != expected_bytes or output.size != expected_bytes:
      print(
          f"[ERROR] Unexpected byte count: golden={golden.size} "
          f"out={output.size} expected={expected_bytes}"
      )
      raise SystemExit(2)
    if not np.array_equal(golden[:PK_STORAGE_BYTES], output[:PK_STORAGE_BYTES]):
        diff = np.nonzero(golden[:PK_STORAGE_BYTES] != output[:PK_STORAGE_BYTES])[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (psti PK raw packed store): idx={idx} "
            f"golden={int(golden[idx])} out={int(output[idx])}"
        )
        raise SystemExit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
