#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: kernels/online-softmax-update
# family: kernels
# target_ops: pto.dma_load, pto.dma_store, pto.vlds, pto.vcmax, pto.vdup, pto.vmax, pto.vexpdif, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv, pto.vsts
# scenarios: online-softmax-update, 16x128-f32, oldmax-oldsum-qk-to-newmax-newsum-expmax-out

import os
import sys

import numpy as np


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        abs_diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        idx = int(np.argmax(abs_diff))
        print(
            f"[ERROR] Mismatch: max diff={float(abs_diff[idx])} at idx={idx} "
            f"(golden={float(golden[idx])}, out={float(output[idx])}, dtype={dtype_np})"
        )
        return False
    return True


def compare_matrix_valid(golden_path, output_path, rows, cols, valid_cols, dtype, eps):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    expected_elems = rows * cols
    if golden.size != expected_elems or output.size != expected_elems:
        print(
            f"[ERROR] Shape mismatch: expected elems={expected_elems}, "
            f"golden={golden.size}, out={output.size}"
        )
        return False
    golden = golden.reshape(rows, cols)
    output = output.reshape(rows, cols)
    if not np.allclose(
        golden[:, :valid_cols],
        output[:, :valid_cols],
        atol=eps,
        rtol=eps,
        equal_nan=True,
    ):
        abs_diff = np.abs(
            golden[:, :valid_cols].astype(np.float64)
            - output[:, :valid_cols].astype(np.float64)
        )
        flat_idx = int(np.argmax(abs_diff))
        row, col = divmod(flat_idx, valid_cols)
        print(
            f"[ERROR] Mismatch in valid region: max diff={float(abs_diff[row, col])} "
            f"at row={row}, col={col} "
            f"(golden={float(golden[row, col])}, out={float(output[row, col])}, dtype={dtype_np})"
        )
        return False
    return True


def main():
    rows = int(np.fromfile("v9.bin", dtype=np.int32)[0])
    seq = int(np.fromfile("v8.bin", dtype=np.int32)[0])
    ok = True
    ok = compare_bin("golden_v4.bin", "v4.bin", np.float32, 1e-4) and ok
    ok = compare_bin("golden_v5.bin", "v5.bin", np.float32, 1e-4) and ok
    ok = compare_bin("golden_v6.bin", "v6.bin", np.float32, 1e-4) and ok
    ok = compare_matrix_valid(
        "golden_v7.bin", "v7.bin", rows, 128, seq, np.float32, 1e-4
    ) and ok
    if not ok:
        print("[ERROR] compare failed")
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
