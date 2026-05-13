#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import struct
from pathlib import Path

import numpy as np


M = 40
N = 64
K = 50
FP_RELU_ELEMS = 128
ALPHA = np.float32(0.25)
VECTOR_ALPHA_PATTERN = np.array([0.125, 0.25, 0.5, 0.75], dtype=np.float32)
SEED = 521


def encode_scale(scale: float) -> np.uint32:
    return np.uint32(struct.unpack("!I", struct.pack("!f", scale))[0])


def scalar_relu(data: np.ndarray) -> np.ndarray:
    return np.where(data >= np.float32(0.0), data, data * ALPHA)


def make_vector_alphas() -> np.ndarray:
    return np.resize(VECTOR_ALPHA_PATTERN, N).astype(np.float32)


def make_vector_relu_params(vector_alphas: np.ndarray) -> np.ndarray:
    payload = np.resize(vector_alphas, FP_RELU_ELEMS).astype(np.float32)
    return np.array([encode_scale(float(alpha)) for alpha in payload], dtype=np.uint32)


def vector_relu(data: np.ndarray, vector_alphas: np.ndarray) -> np.ndarray:
    return np.where(
        data >= np.float32(0.0),
        data,
        data * vector_alphas.reshape(1, N),
    )


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lhs = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float16)
    rhs = rng.uniform(-1.5, 1.5, size=(K, N)).astype(np.float16)
    vector_alphas = make_vector_alphas()
    relu_fp = make_vector_relu_params(vector_alphas)

    lhs32 = lhs.astype(np.float32)
    rhs32 = rhs.astype(np.float32)
    matmul = np.zeros((M, N), dtype=np.float32)
    for k_idx in range(K):
        matmul += lhs32[:, k_idx:k_idx + 1] * rhs32[k_idx:k_idx + 1, :]

    scalar_golden = scalar_relu(matmul).astype(np.float16)
    vector_golden = vector_relu(matmul, vector_alphas).astype(np.float16)
    if np.array_equal(scalar_golden, vector_golden):
        raise AssertionError("vector relu golden must differ from scalar relu golden")

    output_dir.mkdir(parents=True, exist_ok=True)
    lhs.reshape(-1).tofile(output_dir / "v1.bin")
    rhs.reshape(-1).tofile(output_dir / "v2.bin")
    relu_fp.reshape(-1).tofile(output_dir / "v3.bin")
    for index in range(4, 10):
        np.zeros((M, N), dtype=np.float16).reshape(-1).tofile(output_dir / f"v{index}.bin")
        golden = scalar_golden if index in (4, 6, 8) else vector_golden
        golden.reshape(-1).tofile(output_dir / f"golden_v{index}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
