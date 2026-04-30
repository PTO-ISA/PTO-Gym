#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/rearrangement/vsqz-nontrivial-mask
# family: rearrangement
# target_ops: pto.vsqz
# scenarios: predicate-driven-rearrangement, stable-order, nontrivial-mask
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
LANES = 64
BLOCKS = ROWS * COLS // LANES
ACTIVE_POSITIONS = [1, 4, 5, 9, 12, 16, 21, 24, 29, 33, 36, 40, 45, 49, 54, 60]
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-8.0, 8.0, size=(BLOCKS, LANES)).astype(np.float32)
    mask_seed = np.full((BLOCKS, LANES), -1.0, dtype=np.float32)
    golden = np.zeros((BLOCKS, LANES), dtype=np.float32)

    for block in range(BLOCKS):
        for pos in ACTIVE_POSITIONS:
            mask_seed[block, pos] = 1.0
        kept = values[block, ACTIVE_POSITIONS]
        golden[block, :kept.size] = kept

    output_dir.mkdir(parents=True, exist_ok=True)
    values.reshape(-1).tofile(output_dir / "v1.bin")
    mask_seed.reshape(-1).tofile(output_dir / "v2.bin")
    golden.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nontrivial-mask inputs/golden for VPTO micro-op vsqz validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
