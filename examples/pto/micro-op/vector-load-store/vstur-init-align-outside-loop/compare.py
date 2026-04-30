#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vstur-init-align-outside-loop
# family: vector-load-store
# target_ops: pto.vstur
# scenarios: core-f32, full-mask, unaligned, state-update, init-align-outside-loop
# coding=utf-8

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
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        if golden.size:
            g = golden.astype(np.float64, copy=False)
            o = output.astype(np.float64, copy=False)
            abs_diff = np.abs(g - o)
            idx = int(np.argmax(abs_diff))
            diff = float(abs_diff[idx])
            print(
                f"[ERROR] Mismatch: {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                f"(golden={g[idx]}, out={o[idx]}, dtype={dtype_np})"
            )
        else:
            print(f"[ERROR] Mismatch: {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
        return False
    return True


def compare_bin_window(golden_path, output_path, dtype, eps, offset, count):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    try:
        offset = int(offset)
        count = int(count)
    except Exception:
        print(f"[ERROR] Invalid compare window: offset={offset} count={count}")
        return False
    if offset < 0 or count <= 0:
        print(f"[ERROR] Invalid compare window: offset={offset} count={count}")
        return False

    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    end = offset + count
    if golden.size < end or output.size < end:
        print(
            f"[ERROR] Compare window out of range: offset={offset} count={count}, "
            f"golden={golden.size}, out={output.size}"
        )
        return False

    golden_sel = golden[offset:end]
    output_sel = output[offset:end]
    if not np.allclose(golden_sel, output_sel, atol=eps, rtol=eps, equal_nan=True):
        if golden_sel.size:
            g = golden_sel.astype(np.float64, copy=False)
            o = output_sel.astype(np.float64, copy=False)
            abs_diff = np.abs(g - o)
            idx = int(np.argmax(abs_diff))
            diff = float(abs_diff[idx])
            print(
                f"[ERROR] Mismatch (window): {golden_path} vs {output_path}, max diff={diff} "
                f"at idx={offset + idx} (golden={g[idx]}, out={o[idx]}, dtype={dtype_np}, "
                f"offset={offset}, count={count})"
            )
        else:
            print(f"[ERROR] Mismatch (window): {golden_path} vs {output_path}, empty window, dtype={dtype_np}")
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin_window("golden_v2.bin", "v2.bin", np.float32, 0.0001, 1, 8)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
