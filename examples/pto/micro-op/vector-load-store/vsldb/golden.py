#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vsldb
# family: vector-load-store
# target_ops: pto.vsldb
# scenarios: core-f32, full-mask, block-strided-load, block-mask
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
BLOCK_STRIDE = 2
REPEAT_STRIDE = 4
BLOCK_ELEMS = 8
BLOCK_COUNT = 8


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
    v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    flat_in = v1.reshape(-1)
    flat_golden = golden_v2.reshape(-1)
    for blk in range(BLOCK_COUNT):
        src_blk = REPEAT_STRIDE + blk * BLOCK_STRIDE
        flat_golden[blk * BLOCK_ELEMS:(blk + 1) * BLOCK_ELEMS] = flat_in[
            src_blk * BLOCK_ELEMS:(src_blk + 1) * BLOCK_ELEMS
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    golden_v2.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vsldb validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
