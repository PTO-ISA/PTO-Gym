#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/conversion/vcvt-tail-special
# family: conversion
# target_ops: pto.vcvt
# scenarios: f32-to-f16, tail-mask, exceptional-values
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
LOGICAL_ELEMS = 1000


def generate(output_dir: Path, seed: int) -> None:
    del seed
    special = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            np.inf,
            -np.inf,
            np.nan,
            65504.0,
            -65504.0,
            1.0e-8,
            -1.0e-8,
            1.0e-4,
            -1.0e-4,
            123.75,
            -123.75,
            0.33333334,
        ],
        dtype=np.float32,
    )
    flat = np.resize(special, ROWS * COLS).astype(np.float32)
    flat[LOGICAL_ELEMS:] = 0.0
    v1 = flat.reshape(ROWS, COLS)
    v2 = np.zeros((ROWS, COLS), dtype=np.float16)
    golden_flat = np.zeros(ROWS * COLS, dtype=np.float16)

    remaining = LOGICAL_ELEMS
    for offset in range(0, ROWS * COLS, 128):
        lower = flat[offset : offset + 64].astype(np.float16)
        upper = flat[offset + 64 : offset + 128].astype(np.float16)
        merged = np.empty(128, dtype=np.float16)
        merged[0::2] = lower
        merged[1::2] = upper
        active = min(remaining, 128)
        golden_flat[offset : offset + active] = merged[:active]
        remaining = max(remaining - 128, 0)

    golden_v2 = golden_flat.reshape(ROWS, COLS)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    golden_v2.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vcvt-tail-special validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
