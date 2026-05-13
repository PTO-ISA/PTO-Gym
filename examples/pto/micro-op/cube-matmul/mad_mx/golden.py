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
SCALE_BYTES = 64


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


def e8m0_to_f32(bits: np.ndarray) -> np.ndarray:
    return np.exp2(bits.astype(np.int32) - 127).astype(np.float32)


def pack_a_scale(a_scale: np.ndarray) -> np.ndarray:
    packed = np.zeros(SCALE_BYTES, dtype=np.uint8)
    packed[0:32] = a_scale.reshape(-1)
    return packed


def pack_b_scale(b_scale: np.ndarray) -> np.ndarray:
    packed = np.zeros(SCALE_BYTES, dtype=np.uint8)
    packed[0:32] = b_scale.T.reshape(-1)
    return packed


def generate(output_dir: Path) -> None:
    # Values are exactly representable in FP8 E4M3: 0.5, 1.0, 2.0 and -1.0.
    a_codes = np.array([0x30, 0x38, 0x40, 0xB8], dtype=np.uint8)
    m_idx = np.arange(M).reshape(M, 1)
    k_idx = np.arange(K).reshape(1, K)
    a_matrix = a_codes[(m_idx * 3 + k_idx * 5) % a_codes.size]
    b_matrix = np.full((K, N), 0x38, dtype=np.uint8)

    # E8M0 scale is 2^(byte - 127). The two K/32 groups use different scales,
    # and A scales vary by M so the test catches incorrect scale grouping.
    a_scale_matrix = np.where(
        (np.arange(M).reshape(M, 1) + np.arange(2)) % 2 == 0, 127, 128
    ).astype(np.uint8)
    b_scale_matrix = np.array([[126], [127]], dtype=np.uint8).repeat(N, axis=1)
    a = a_matrix.reshape(-1).astype(np.uint8)
    b = b_matrix.reshape(-1).astype(np.uint8)
    a_scale = pack_a_scale(a_scale_matrix)
    b_scale = pack_b_scale(b_scale_matrix)
    c = np.zeros((M, N), dtype=np.float32)

    a_f32 = fp8_e4m3_to_f32(a_matrix)
    b_f32 = fp8_e4m3_to_f32(b_matrix)
    golden_c = np.zeros((M, N), dtype=np.float32)
    a_scale_f32 = e8m0_to_f32(a_scale_matrix)
    b_scale_f32 = e8m0_to_f32(b_scale_matrix)
    for group in range(K // 32):
        k_slice = slice(group * 32, (group + 1) * 32)
        scaled_a = a_f32[:, k_slice] * a_scale_f32[:, group : group + 1]
        scaled_b = b_f32[k_slice, :] * b_scale_f32[group : group + 1, :]
        golden_c += scaled_a @ scaled_b

    output_dir.mkdir(parents=True, exist_ok=True)
    a.tofile(output_dir / "v1.bin")
    b.tofile(output_dir / "v2.bin")
    a_scale.tofile(output_dir / "v4.bin")
    b_scale.tofile(output_dir / "v5.bin")
    c.reshape(-1).tofile(output_dir / "v3.bin")
    golden_c.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
