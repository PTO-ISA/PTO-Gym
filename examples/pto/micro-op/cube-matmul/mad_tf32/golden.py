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


def tf32_round_even(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(13)) & np.uint32(1)
    rounded = bits + np.uint32(0x00000FFF) + lsb
    return (rounded & np.uint32(0xFFFFE000)).view(np.float32)


def generate(output_dir: Path) -> None:
    row = np.arange(M, dtype=np.float32).reshape(M, 1)
    col = np.arange(K, dtype=np.float32).reshape(1, K)
    a_base = (((row * 11 + col * 3) % 31) - 15).astype(np.float32) / 7.0
    a_perturb = (((row * 5 + col * 9) % 17) + 1).astype(np.float32)
    a = a_base + a_perturb * np.float32(2.0 ** -13)
    k_idx = np.arange(K, dtype=np.float32).reshape(K, 1)
    n_idx = np.arange(N, dtype=np.float32).reshape(1, N)
    b_base = (((k_idx * 5 - n_idx * 13) % 37) - 18).astype(np.float32) / 9.0
    b_perturb = (((k_idx * 7 + n_idx * 3) % 19) + 1).astype(np.float32)
    b = b_base - b_perturb * np.float32(2.0 ** -13)
    c_tf32 = np.zeros((M, N), dtype=np.float32)
    c_plain = np.zeros((M, N), dtype=np.float32)
    golden_tf32 = tf32_round_even(a) @ tf32_round_even(b)
    plain_fp32_c = a @ b
    max_tf32_delta = float(np.max(np.abs(plain_fp32_c - golden_tf32)))
    print(f"[INFO] max plain-fp32-vs-tf32 delta: {max_tf32_delta}")

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    c_tf32.reshape(-1).tofile(output_dir / "v3.bin")
    c_plain.reshape(-1).tofile(output_dir / "v4.bin")
    golden_tf32.reshape(-1).tofile(output_dir / "golden_v3.bin")
    plain_fp32_c.reshape(-1).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
