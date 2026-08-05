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

THREADS = 32
BASE64_A = 0x1122334400000000
BASE64_B = 0x5566778800000000


def main() -> None:
    actual_i32 = np.fromfile("output_i32.bin", dtype=np.int32)
    actual_i64 = np.fromfile("output_i64.bin", dtype=np.int64)
    tids = np.arange(THREADS, dtype=np.int32)
    expected_i32 = np.empty(2 * THREADS, dtype=np.int32)
    expected_i32[0::2] = tids + 200
    expected_i32[1::2] = tids + 100
    tids_i64 = tids.astype(np.int64)
    expected_i64 = np.empty(2 * THREADS, dtype=np.int64)
    expected_i64[0::2] = BASE64_B + tids_i64
    expected_i64[1::2] = BASE64_A + tids_i64

    for name, actual, expected in (
        ("i32", actual_i32, expected_i32),
        ("i64", actual_i64, expected_i64),
    ):
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            mismatch = np.flatnonzero(actual != expected)
            index = int(mismatch[0]) if mismatch.size else -1
            print(
                f"[ERROR] {name} mismatch at index {index}: "
                f"actual={actual[index] if index >= 0 else actual.shape}, "
                f"expected={expected[index] if index >= 0 else expected.shape}"
            )
            sys.exit(1)
    print(f"[PASS] all {THREADS} lanes preserved the i32 and i64 swaps")


if __name__ == "__main__":
    main()
