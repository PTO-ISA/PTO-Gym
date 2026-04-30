#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19
SCALE = np.float32(1.5)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    wide = values.astype(np.float32, copy=False).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((wide >> 16) & np.uint32(1))
    return ((wide + rounding) >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1_f32 = rng.uniform(-4.0, 4.0, size=ELEMS).astype(np.float32)
    v1 = f32_to_bf16_bits(v1_f32)
    v2 = np.zeros(ELEMS, dtype=np.uint16)
    scalar_bits = f32_to_bf16_bits(np.array([SCALE], dtype=np.float32))[0]
    scalar = bf16_bits_to_f32(np.array([scalar_bits], dtype=np.uint16))[0]
    golden_v2 = f32_to_bf16_bits(bf16_bits_to_f32(v1) + scalar)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
