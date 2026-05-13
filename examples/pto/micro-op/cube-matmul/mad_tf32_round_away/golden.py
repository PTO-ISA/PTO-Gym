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
K = 16


def tf32_round_away(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x00001000)
    return (rounded & np.uint32(0xFFFFE000)).view(np.float32)


def tf32_round_even(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(13)) & np.uint32(1)
    rounded = bits + np.uint32(0x00000FFF) + lsb
    return (rounded & np.uint32(0xFFFFE000)).view(np.float32)


def generate(output_dir: Path) -> None:
    row = np.arange(M, dtype=np.uint32).reshape(M, 1)
    col = np.arange(K, dtype=np.uint32).reshape(1, K)
    a_sign = ((row + col) & np.uint32(1)) << np.uint32(31)
    a_mant = ((row * np.uint32(29) + col * np.uint32(37)) % np.uint32(512))
    a_bits = a_sign | np.uint32(0x3F800000) | (a_mant << np.uint32(13)) | np.uint32(0x1000)
    a = a_bits.astype(np.uint32).view(np.float32)

    k_idx = np.arange(K, dtype=np.uint32).reshape(K, 1)
    n_idx = np.arange(N, dtype=np.uint32).reshape(1, N)
    b_sign = ((k_idx * np.uint32(3) + n_idx) & np.uint32(1)) << np.uint32(31)
    b_mant = ((k_idx * np.uint32(41) + n_idx * np.uint32(11)) % np.uint32(512))
    b_bits = b_sign | np.uint32(0x3F800000) | (b_mant << np.uint32(13)) | np.uint32(0x1000)
    b = b_bits.astype(np.uint32).view(np.float32)
    c = np.zeros((M, N), dtype=np.float32)
    golden_c = tf32_round_away(a) @ tf32_round_away(b)
    round_even_c = tf32_round_even(a) @ tf32_round_even(b)
    plain_fp32_c = a @ b
    max_tf32_delta = float(np.max(np.abs(plain_fp32_c - golden_c)))
    max_round_mode_delta = float(np.max(np.abs(round_even_c - golden_c)))
    print(f"[INFO] max plain-fp32-vs-tf32 delta: {max_tf32_delta}")
    print(f"[INFO] max round-even-vs-round-away delta: {max_round_mode_delta}")

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    c.reshape(-1).tofile(output_dir / "v3.bin")
    golden_c.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
