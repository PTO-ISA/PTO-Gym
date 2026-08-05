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

F32_CHECKS = (
    ("f8e4m3fn-to-f32", "golden_v6_f8e4_f32_out.bin", "v6_f8e4_f32_out.bin"),
    ("f8e5m2-to-f32", "golden_v7_f8e5_f32_out.bin", "v7_f8e5_f32_out.bin"),
    ("hif8-to-f32", "golden_v8_hif8_f32_out.bin", "v8_hif8_f32_out.bin"),
)

BF16_CHECKS = (
    ("f4e1m2x2-to-bf16", "golden_v9_f4e1_bf16_out.bin", "v9_f4e1_bf16_out.bin"),
    ("f4e2m1x2-to-bf16", "golden_v10_f4e2_bf16_out.bin", "v10_f4e2_bf16_out.bin"),
)

BYTE_CHECKS = (
    ("f32-to-f8e4m3fn", "golden_v16_f8e4_out.bin", "v16_f8e4_out.bin"),
    ("f32-to-f8e5m2", "golden_v17_f8e5_out.bin", "v17_f8e5_out.bin"),
    ("f32-to-hif8", "golden_v18_hif8_out.bin", "v18_hif8_out.bin"),
    ("bf16-to-f4e1m2x2", "golden_v19_f4e1_out.bin", "v19_f4e1_out.bin"),
    ("bf16-to-f4e2m1x2", "golden_v20_f4e2_out.bin", "v20_f4e2_out.bin"),
)


def compare_f32(label, golden_path, output_path):
    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    if golden.shape != output.shape:
        print(f"[ERROR] {label}: shape mismatch golden={golden.shape} output={output.shape}")
        return False

    golden_bits = golden.view(np.uint32)
    output_bits = output.view(np.uint32)
    both_nan = np.isnan(golden) & np.isnan(output)
    same = (golden_bits == output_bits) | both_nan
    if np.all(same):
        return True

    idx = int(np.nonzero(~same)[0][0])
    print(
        f"[ERROR] {label}: mismatch idx={idx} "
        f"golden={golden[idx]} bits=0x{int(golden_bits[idx]):08x} "
        f"output={output[idx]} bits=0x{int(output_bits[idx]):08x}"
    )
    return False


def compare_u16(label, golden_path, output_path):
    golden = np.fromfile(golden_path, dtype=np.uint16)
    output = np.fromfile(output_path, dtype=np.uint16)
    if golden.shape != output.shape:
        print(f"[ERROR] {label}: shape mismatch golden={golden.shape} output={output.shape}")
        return False
    if np.array_equal(golden, output):
        return True

    idx = int(np.nonzero(golden != output)[0][0])
    print(
        f"[ERROR] {label}: mismatch idx={idx} "
        f"golden=0x{int(golden[idx]):04x} output=0x{int(output[idx]):04x}"
    )
    return False


def compare_u8(label, golden_path, output_path):
    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    if golden.shape != output.shape:
        print(f"[ERROR] {label}: shape mismatch golden={golden.shape} output={output.shape}")
        return False
    if np.array_equal(golden, output):
        return True

    idx = int(np.nonzero(golden != output)[0][0])
    print(
        f"[ERROR] {label}: byte mismatch idx={idx} "
        f"golden=0x{int(golden[idx]):02x} output=0x{int(output[idx]):02x}"
    )
    return False


def main():
    ok = True
    for check in F32_CHECKS:
        ok = compare_f32(*check) and ok
    for check in BF16_CHECKS:
        ok = compare_u16(*check) and ok
    for check in BYTE_CHECKS:
        ok = compare_u8(*check) and ok
    if not ok:
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
