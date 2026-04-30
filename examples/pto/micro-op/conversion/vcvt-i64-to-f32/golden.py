#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/conversion/vcvt-i64-to-f32
# family: conversion
# target_ops: pto.dma_load, pto.dma_store, pto.vcvt, pto.vsts
# scenarios: i64-dma-roundtrip, i64-to-f32, signed-input, rounded, part-even-low-half

import argparse
from pathlib import Path

import numpy as np


INPUT_ELEMS = 1024
OUTPUT_ELEMS = 512
ROUNDTRIP_ELEMS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    edge = np.array(
        [
            -(1 << 31),
            -(1 << 24) - 3,
            -(1 << 24) - 1,
            -(1 << 24),
            -(1 << 24) + 1,
            -65537,
            -32769,
            -32768,
            -1,
            0,
            1,
            32767,
            32768,
            65537,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 24) + 3,
            (1 << 31) - 2,
            (1 << 31) - 1,
        ],
        dtype=np.int32,
    )
    base = rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max,
                        size=INPUT_ELEMS, dtype=np.int32)
    base[: edge.size] = edge
    v1 = base.astype(np.int64)
    v2 = np.zeros(OUTPUT_ELEMS, dtype=np.float32)
    v3 = np.zeros(ROUNDTRIP_ELEMS, dtype=np.int64)
    golden_v2 = np.concatenate(
        [base[offset : offset + 16] for offset in range(0, INPUT_ELEMS, 32)]
    ).astype(np.float32)
    golden_v3 = v1.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    v3.tofile(output_dir / "v3.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")
    golden_v3.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vcvt-i64-to-f32 validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
