#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare output buffers against golden for merged vector ldg/stg test.

Arithmetic types use exact match (np.array_equal) for integer types,
and np.allclose for floating types (no tolerance for f32x2, f16x2, bf16x2).
"""

import os
import sys

import numpy as np

STRICT = os.getenv("COMPARE_STRICT", "1") != "0"


def _check_exact(name, golden_path, out_path, dtype):
    """Exact match comparison."""
    golden = np.fromfile(golden_path, dtype=dtype)
    out = np.fromfile(out_path, dtype=dtype)
    if golden.shape != out.shape:
        print(f"[ERROR] {name} shape mismatch: golden {golden.shape} vs out {out.shape}")
        return False
    ok = np.array_equal(golden, out)
    if not ok:
        idxs = np.nonzero(golden != out)[0]
        idx = int(idxs[0]) if idxs.size else 0
        print(f"[ERROR] {name} mismatch at idx={idx}: golden={golden[idx]}, out={out[idx]}")
        if idxs.size > 1:
            print(f"       ({idxs.size} total mismatches, first shown)")
    return ok


def _check_allclose(name, golden_path, out_path, dtype):
    """All-close comparison for floating types."""
    golden = np.fromfile(golden_path, dtype=dtype)
    out = np.fromfile(out_path, dtype=dtype)
    if golden.shape != out.shape:
        print(f"[ERROR] {name} shape mismatch: golden {golden.shape} vs out {out.shape}")
        return False
    ok = np.allclose(golden, out, rtol=0, atol=0, equal_nan=True)
    if not ok:
        idxs = np.nonzero(~np.isclose(golden, out, rtol=0, atol=0, equal_nan=True))[0]
        idx = int(idxs[0]) if idxs.size else 0
        print(f"[ERROR] {name} mismatch at idx={idx}: golden={golden[idx]}, out={out[idx]}")
        if idxs.size > 1:
            print(f"       ({idxs.size} total mismatches, first shown)")
    return ok


def main():
    all_ok = True

    # Arithmetic: exact match for all (f32x2 uses allclose with atol=0)
    all_ok &= _check_allclose("f32x2", "golden_f32x2.bin", "f32x2.bin", np.float32)
    all_ok &= _check_allclose("f16x2", "golden_f16x2.bin", "f16x2.bin", np.float16)
    all_ok &= _check_exact("bf16x2", "golden_bf16x2.bin", "bf16x2.bin", np.uint16)
    all_ok &= _check_exact("i16x2", "golden_i16x2.bin", "i16x2.bin", np.int16)
    all_ok &= _check_exact("i32x2", "golden_i32x2.bin", "i32x2.bin", np.int32)

    # Copy: exact match for raw payload
    all_ok &= _check_exact("hif8x2", "golden_hif8x2.bin", "hif8x2.bin", np.uint16)
    all_ok &= _check_exact("i8x2", "golden_i8x2.bin", "i8x2.bin", np.uint8)
    all_ok &= _check_exact("fp8e4x2", "golden_fp8e4x2.bin", "fp8e4x2.bin", np.uint16)
    all_ok &= _check_exact("fp8e5x2", "golden_fp8e5x2.bin", "fp8e5x2.bin", np.uint16)

    if all_ok:
        print("[INFO] all checks passed")
    else:
        if STRICT:
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")


if __name__ == "__main__":
    main()
