#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/rearrangement/vpack-lower
# family: rearrangement
# target_ops: pto.vpack
# scenarios: narrowing, lower-half-placement, zero-fill-upper-half, post-pack-consumer
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
ELEMS = ROWS * COLS
CHUNK = 64
OUTPUT_ELEMS = ELEMS * 2
SEED = 19
BIAS = np.uint16(1)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-(1 << 20), 1 << 20, size=ELEMS, dtype=np.int32)
    v2 = np.zeros(OUTPUT_ELEMS, dtype=np.uint16)
    golden_v2 = np.zeros(OUTPUT_ELEMS, dtype=np.uint16)

    narrowed = v1.astype(np.uint16, copy=False)
    for chunk_base in range(0, ELEMS, CHUNK):
        chunk = narrowed[chunk_base : chunk_base + CHUNK]
        out_base = (chunk_base // CHUNK) * (CHUNK * 2)
        golden_v2[out_base : out_base + CHUNK] = (
            chunk.astype(np.uint32) + int(BIAS)
        ).astype(np.uint16)
        golden_v2[out_base + CHUNK : out_base + 2 * CHUNK] = BIAS

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vpack-lower validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
