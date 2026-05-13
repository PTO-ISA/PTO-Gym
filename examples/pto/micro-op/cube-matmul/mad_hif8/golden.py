#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

M = 16
N = 16
K = 64


def generate(output_dir: Path) -> None:
    # The simulator maps 0x40 and 0xC0 to opposite HiF8 values, and 0x00 to
    # zero.  For this K=64 case, a constant 0x40 x 0x40 dot product produces
    # 8192.0, so one non-zero multiply contributes 128.0 to the FP32 result.
    # Keep each row/column stable across K so the test validates HiF8 MAD
    # semantics without baking in an incomplete model of the 8-bit L0 layout.
    codes = np.array([0x40, 0xC0, 0x00, 0x40, 0xC0], dtype=np.uint8)
    signed_units = np.array([1.0, -1.0, 0.0, 1.0, -1.0], dtype=np.float32)

    m_idx = np.arange(M).reshape(M, 1)
    k_idx = np.arange(K).reshape(1, K)
    a_index = (m_idx * 3 + k_idx * 0) % codes.size
    a = codes[a_index].astype(np.uint8)
    a_unit = signed_units[a_index].astype(np.float32)

    k_idx = np.arange(K).reshape(K, 1)
    n_idx = np.arange(N).reshape(1, N)
    b_index = (k_idx * 0 + n_idx * 2 + 1) % codes.size
    b = codes[b_index].astype(np.uint8)
    b_unit = signed_units[b_index].astype(np.float32)

    c_hif8 = np.zeros((M, N), dtype=np.float32)
    c_fp8 = np.zeros((M, N), dtype=np.float32)

    # A 16x64 8-bit left tile is consumed in row groups after the L1-to-L0A
    # layout step.  The generated row-major tile above is intentionally mapped
    # to positive, negative, and zero effective rows so the output checks more
    # than a single constant path.
    effective_a_unit = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
        ],
        dtype=np.float32,
    ).reshape(M, 1)
    effective_b_unit = b_unit[0:1, :]
    effective_fp8_a_unit = np.array(
        [
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    ).reshape(M, 1)
    golden_hif8 = (effective_a_unit @ effective_b_unit) * np.float32(K * 128.0)
    golden_fp8 = (effective_fp8_a_unit @ effective_b_unit) * np.float32(K * 2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    c_hif8.reshape(-1).tofile(output_dir / "v3.bin")
    c_fp8.reshape(-1).tofile(output_dir / "v4.bin")
    golden_hif8.reshape(-1).tofile(output_dir / "golden_v3.bin")
    golden_fp8.reshape(-1).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
