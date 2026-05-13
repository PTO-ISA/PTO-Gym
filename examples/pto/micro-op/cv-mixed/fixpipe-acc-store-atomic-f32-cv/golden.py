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
ATOMIC_ADD_INIT = np.float32(1.25)
ATOMIC_DELTA = np.float32(0.5)
SEED = 419


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lhs = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float16)
    rhs = rng.uniform(-1.5, 1.5, size=(K, N)).astype(np.float16)
    lhs32 = lhs.astype(np.float32)
    rhs32 = rhs.astype(np.float32)
    matmul = np.zeros((M, N), dtype=np.float32)
    for k_idx in range(K):
        matmul += lhs32[:, k_idx:k_idx + 1] * rhs32[k_idx:k_idx + 1, :]

    plain_init = np.zeros((M, N), dtype=np.float32)
    atomic_add_init = np.full((M, N), ATOMIC_ADD_INIT, dtype=np.float32)
    parity = ((np.arange(M * N, dtype=np.int32).reshape(M, N) & 1) * 2 - 1).astype(np.float32)
    atomic_max_init = matmul + parity * ATOMIC_DELTA
    atomic_min_init = matmul - parity * ATOMIC_DELTA
    atomic_add_golden = atomic_add_init + matmul
    atomic_max_golden = np.maximum(atomic_max_init, matmul)
    atomic_min_golden = np.minimum(atomic_min_init, matmul)

    output_dir.mkdir(parents=True, exist_ok=True)
    lhs.reshape(-1).tofile(output_dir / "v1.bin")
    rhs.reshape(-1).tofile(output_dir / "v2.bin")
    plain_init.reshape(-1).tofile(output_dir / "v3.bin")
    atomic_add_init.reshape(-1).tofile(output_dir / "v4.bin")
    atomic_max_init.reshape(-1).tofile(output_dir / "v5.bin")
    atomic_min_init.reshape(-1).tofile(output_dir / "v6.bin")
    matmul.reshape(-1).tofile(output_dir / "golden_v3.bin")
    atomic_add_golden.reshape(-1).tofile(output_dir / "golden_v4.bin")
    atomic_max_golden.reshape(-1).tofile(output_dir / "golden_v5.bin")
    atomic_min_golden.reshape(-1).tofile(output_dir / "golden_v6.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
