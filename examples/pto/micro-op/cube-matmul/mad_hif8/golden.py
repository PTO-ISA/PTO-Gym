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
M0 = 16
K0 = 32
N0 = 16


def pack_lhs_cube_fractal(matrix: np.ndarray) -> np.ndarray:
    m, k = matrix.shape
    assert m % M0 == 0 and k % K0 == 0
    return matrix.reshape(m // M0, M0, k // K0, K0).transpose(
        2, 0, 1, 3
    )


def pack_rhs_cube_fractal(matrix: np.ndarray) -> np.ndarray:
    k, n = matrix.shape
    assert k % K0 == 0 and n % N0 == 0
    return matrix.reshape(k // K0, K0, n // N0, N0).transpose(
        0, 2, 1, 3
    )


def fp8_e4m3_to_f32(bits: np.ndarray) -> np.ndarray:
    raw = bits.astype(np.uint8)
    sign = np.where((raw & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((raw >> 3) & 0x0F).astype(np.int32)
    mantissa = (raw & 0x07).astype(np.float32)
    normal = exponent != 0
    value = np.where(
        normal,
        (1.0 + mantissa / 8.0) * np.exp2(exponent - 7),
        (mantissa / 8.0) * np.exp2(-6),
    ).astype(np.float32)
    return sign * value


def generate(output_dir: Path) -> None:
    # This kernel stages GM->L1 with raw byte copies, so v1/v2.bin must already
    # be in the cube-fractal layout expected by mte_l1_l0a/l0b.
    codes = np.array([0x40, 0xC0, 0x00, 0x40, 0xC0], dtype=np.uint8)
    signed_units = np.array([1.0, -1.0, 0.0, 1.0, -1.0], dtype=np.float32)

    m_idx = np.arange(M).reshape(M, 1)
    k_idx = np.arange(K).reshape(1, K)
    a_index = (m_idx * 3 + k_idx * 0) % codes.size
    a_logical = codes[a_index].astype(np.uint8)
    a_unit = signed_units[a_index].astype(np.float32)

    k_idx = np.arange(K).reshape(K, 1)
    n_idx = np.arange(N).reshape(1, N)
    b_index = (k_idx * 0 + n_idx * 2 + 1) % codes.size
    b_logical = codes[b_index].astype(np.uint8)
    b_unit = signed_units[b_index].astype(np.float32)

    a = pack_lhs_cube_fractal(a_logical).reshape(-1).astype(np.uint8)
    b = pack_rhs_cube_fractal(b_logical).reshape(-1).astype(np.uint8)
    c_hif8 = np.zeros((M, N), dtype=np.float32)
    c_fp8 = np.zeros((M, N), dtype=np.float32)

    golden_hif8 = (a_unit @ b_unit).astype(np.float32) * np.float32(128.0)
    a_fp8 = fp8_e4m3_to_f32(a_logical)
    b_fp8 = fp8_e4m3_to_f32(b_logical)
    golden_fp8 = (a_fp8 @ b_fp8).astype(np.float32)

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
