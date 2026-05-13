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
FP_QUANT_ELEMS = 64
FP_TRANSPORT_ELEMS = FP_QUANT_ELEMS * 2
CLIP_MAX = np.float16(8.0)
SEED = 97


def extract_quant_params(quant: np.uint64) -> tuple[float, int, int]:
    value = int(quant)
    m1_bits = (value >> 13) & 0x7FFFF
    offset = (value >> 37) & 0x1FF
    sign = (value >> 46) & 0x1

    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    m1 = ((-1) ** sign_bit) * (1 + mantissa / 1024.0) * (2 ** (exponent - 127))
    return m1, offset, sign


def qf322f16_pre(data: np.ndarray, quant: np.ndarray) -> np.ndarray:
    result = np.zeros(data.shape, dtype=np.float16)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            m1, _, _ = extract_quant_params(quant[col])
            scaled = data[row, col].astype(np.float32) * np.float32(m1)
            result[row, col] = np.clip(
                scaled,
                np.finfo(np.float16).min,
                np.finfo(np.float16).max,
            ).astype(np.float16)
    return result


def make_vector_quant_params(n: int) -> np.ndarray:
    scales = (np.arange(n, dtype=np.float32) % np.float32(4.0)) + np.float32(1.0)
    encoded = scales.astype(np.uint64)
    for idx, scale in enumerate(scales):
        encoded[idx] = struct.unpack("!I", struct.pack("!f", float(scale)))[0]
    return np.frombuffer(encoded, np.uint64)


def generate(output_dir: Path, seed: int) -> None:
    a = (np.arange(M * K, dtype=np.float32).reshape(M, K) * np.float32(0.01) +
         np.float32(0.5)).astype(np.float16)
    b = (np.arange(K * N, dtype=np.float32).reshape(K, N) * np.float32(0.005) +
         np.float32(0.25)).astype(np.float16)
    fp = make_vector_quant_params(FP_QUANT_ELEMS)
    matmul = np.zeros((M, N), dtype=np.float32)
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    for k_idx in range(K):
        matmul += a32[:, k_idx:k_idx + 1] * b32[k_idx:k_idx + 1, :]
    golden_quant = qf322f16_pre(matmul, fp)
    golden_clip = np.minimum(golden_quant, CLIP_MAX).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    fp.view(np.uint32).reshape(FP_TRANSPORT_ELEMS).tofile(output_dir / "v3.bin")
    mapping = {
        4: golden_quant,
        5: golden_clip,
        6: golden_quant,
        7: golden_clip,
        8: golden_quant,
        9: golden_clip,
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
