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

CHECKS = (
    ("f8e4m3", "golden_v6_f8e4_out.bin", "v6_f8e4_out.bin"),
    ("f8e5m2", "golden_v7_f8e5_out.bin", "v7_f8e5_out.bin"),
    ("hif8", "golden_v8_hif8_out.bin", "v8_hif8_out.bin"),
    ("f4e1m2x2", "golden_v9_f4e1_out.bin", "v9_f4e1_out.bin"),
    ("f4e2m1x2", "golden_v10_f4e2_out.bin", "v10_f4e2_out.bin"),
)


def compare_bin(label, golden_path, output_path):
    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    if golden.shape != output.shape:
        print(f"[ERROR] {label}: shape mismatch golden={golden.shape} output={output.shape}")
        return False
    if np.array_equal(golden, output):
        return True
    diff = np.nonzero(golden != output)[0]
    idx = int(diff[0])
    print(
        f"[ERROR] {label}: byte mismatch idx={idx} "
        f"golden=0x{int(golden[idx]):02x} output=0x{int(output[idx]):02x}"
    )
    return False


def main():
    if not all(compare_bin(*check) for check in CHECKS):
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
