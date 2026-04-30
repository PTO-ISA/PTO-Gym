#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/rearrangement/vusqz
# family: rearrangement
# target_ops: pto.vusqz
# scenarios: predicate-driven-rearrangement, prefix-count

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
LANES = 64
BLOCKS = ROWS * COLS // LANES
ACTIVE_PER_BLOCK = 16
SEED = 19


def build_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.zeros((BLOCKS, LANES), dtype=np.int32)
    mask_seed = np.full((BLOCKS, LANES), -1.0, dtype=np.float32)
    out = np.zeros((BLOCKS, LANES), dtype=np.int32)

    for block in range(BLOCKS):
        src[block] = np.arange(block * 100 - 31, block * 100 - 31 + LANES, dtype=np.int32)
        mask_seed[block, :ACTIVE_PER_BLOCK] = 1.0
        active_count = 0
        out[block, 0] = 0
        for lane in range(1, LANES):
            if mask_seed[block, lane - 1] > 0.0:
                active_count += 1
            out[block, lane] = active_count

    return src.reshape(ROWS, COLS), mask_seed.reshape(ROWS, COLS), out.reshape(ROWS, COLS)


def generate(output_dir: Path) -> None:
    src, mask_seed, out = build_case()
    output_dir.mkdir(parents=True, exist_ok=True)
    src.reshape(-1).tofile(output_dir / "v1.bin")
    mask_seed.reshape(-1).tofile(output_dir / "v2.bin")
    out.reshape(-1).tofile(output_dir / "golden_v3.bin")
    np.zeros_like(out.reshape(-1)).tofile(output_dir / "v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vusqz prefix-count inputs/golden.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    del args.seed
    generate(args.output_dir)


if __name__ == "__main__":
    main()
