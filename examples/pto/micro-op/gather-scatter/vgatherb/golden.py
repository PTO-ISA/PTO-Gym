#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/gather-scatter/vgatherb
# family: gather-scatter
# target_ops: pto.vgatherb
# scenarios: core-f32, full-mask, block-gather, aligned-base, load-effect-validation, no-alias
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
BLOCK_FLOATS = 8
BLOCKS_PER_ITER = 8
ITER_ELEMS = 64
SEED = 19
OUT_SENTINEL = np.float32(-123.25)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    flat = rng.uniform(-8.0, 8.0, size=(ROWS * COLS,)).astype(np.float32)
    blocks = flat.reshape(-1, BLOCK_FLOATS)
    offsets = np.zeros((ROWS * COLS,), dtype=np.int32)
    gathered = np.full((ROWS * COLS,), OUT_SENTINEL, dtype=np.float32)

    for chunk in range((ROWS * COLS) // ITER_ELEMS):
        block_ids = ((np.arange(BLOCKS_PER_ITER, dtype=np.int32) + chunk * 11) * 7 + 3) % blocks.shape[0]
        offsets[chunk * ITER_ELEMS:chunk * ITER_ELEMS + BLOCKS_PER_ITER] = block_ids * 32
        gathered[chunk * ITER_ELEMS:(chunk + 1) * ITER_ELEMS] = blocks[block_ids].reshape(-1)

    v1 = flat.reshape(ROWS, COLS)
    v2 = offsets.reshape(ROWS, COLS)
    v3 = np.full((ROWS, COLS), OUT_SENTINEL, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    v3.reshape(-1).tofile(output_dir / "v3.bin")
    gathered.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vgatherb validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
