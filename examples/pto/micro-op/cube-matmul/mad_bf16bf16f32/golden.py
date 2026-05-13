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


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = values.astype(np.float32, copy=False)
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def generate(output_dir: Path) -> None:
    row = np.arange(M, dtype=np.float32).reshape(M, 1)
    col = np.arange(K, dtype=np.float32).reshape(1, K)
    a_f32 = (((row * 5 + col * 3) % 23) - 11) / 8.0
    k_idx = np.arange(K, dtype=np.float32).reshape(K, 1)
    n_idx = np.arange(N, dtype=np.float32).reshape(1, N)
    b_f32 = (((k_idx * 2 - n_idx * 7) % 29) - 14) / 9.0
    a = to_bf16_bits(a_f32)
    b = to_bf16_bits(b_f32)
    c = np.zeros((M, N), dtype=np.float32)
    golden_c = bf16_bits_to_f32(a).astype(np.float32) @ bf16_bits_to_f32(b).astype(np.float32)

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
