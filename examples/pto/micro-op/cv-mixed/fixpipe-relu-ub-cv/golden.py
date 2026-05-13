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


M = 40
N = 64
K = 50
CLIP_MAX = np.float16(8.0)
SEED = 211


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lhs = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float16)
    rhs = rng.uniform(-1.5, 1.5, size=(K, N)).astype(np.float16)
    lhs32 = lhs.astype(np.float32)
    rhs32 = rhs.astype(np.float32)
    matmul = np.zeros((M, N), dtype=np.float32)
    for k_idx in range(K):
        matmul += lhs32[:, k_idx:k_idx + 1] * rhs32[k_idx:k_idx + 1, :]
    relu = np.maximum(matmul, np.float32(0.0)).astype(np.float16)
    clipped = np.minimum(matmul.astype(np.float16), CLIP_MAX).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    lhs.reshape(-1).tofile(output_dir / "v1.bin")
    rhs.reshape(-1).tofile(output_dir / "v2.bin")
    mapping = {
        3: relu,
        4: clipped,
        5: relu,
        6: clipped,
        7: relu,
        8: clipped,
    }
    for index, golden in mapping.items():
        np.zeros((M, N), dtype=np.float16).reshape(-1).tofile(output_dir / f"v{index}.bin")
        golden.reshape(-1).tofile(output_dir / f"golden_v{index}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
