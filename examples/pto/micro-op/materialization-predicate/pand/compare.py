#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/materialization-predicate/pand
# family: materialization-predicate
# target_ops: pto.pand
# scenarios: predicate-transform, logical-and
# coding=utf-8

import os
import sys

import numpy as np


EXPECTED_WORDS = 32
PREFIX_WORDS = 8


def compare_words(golden_path, output_path):
    if not os.path.exists(output_path) or not os.path.exists(golden_path):
        print(f"[ERROR] Missing file: golden={golden_path} out={output_path}")
        return False
    golden = np.fromfile(golden_path, dtype=np.uint32)
    output = np.fromfile(output_path, dtype=np.uint32)
    if golden.size != EXPECTED_WORDS or output.size != EXPECTED_WORDS:
        print(
            f"[ERROR] Unexpected word count: golden={golden.size} "
            f"out={output.size} expected={EXPECTED_WORDS}"
        )
        return False
    golden = golden[:PREFIX_WORDS]
    output = output[:PREFIX_WORDS]
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (packed predicate words): idx={idx} "
            f"golden={int(golden[idx])} out={int(output[idx])}"
        )
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_words("golden_v1.bin", "v1.bin")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
