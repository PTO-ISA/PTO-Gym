#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/materialization-predicate/vdup-lane
# family: materialization-predicate
# target_ops: pto.vdup
# scenarios: core-f32, vector-input, lowest-highest
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    src = rng.normal(loc=1.0, scale=3.0, size=(ROWS, COLS)).astype(np.float32)

    src_flat = src.reshape(-1)
    low_flat = np.empty_like(src_flat)
    high_flat = np.empty_like(src_flat)
    block = 64
    for begin in range(0, src_flat.size, block):
        chunk = src_flat[begin : begin + block]
        low_flat[begin : begin + block] = chunk[0]
        high_flat[begin : begin + block] = chunk[-1]

    low = low_flat.reshape(src.shape)
    high = high_flat.reshape(src.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.reshape(-1).tofile(output_dir / "src.bin")
    low.reshape(-1).tofile(output_dir / "golden_low.bin")
    high.reshape(-1).tofile(output_dir / "golden_high.bin")
    np.zeros_like(src.reshape(-1)).tofile(output_dir / "out_low.bin")
    np.zeros_like(src.reshape(-1)).tofile(output_dir / "out_high.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO vector-input vdup validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
