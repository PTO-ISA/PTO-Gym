#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/dsa-sfu/vexpdif-boundary
# family: dsa-sfu
# target_ops: pto.vexpdif
# scenarios: core-f32, fused-expdiff, exceptional-values, floating-overflow-underflow
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
def generate(output_dir: Path, seed: int) -> None:
    del seed
    src_pattern = np.array(
        [
            0.0, 88.0, -120.0, np.nan, np.inf, -np.inf, 1.0, -1.0,
            90.0, -90.0, 50.0, -50.0, 3.0, -3.0, 10.0, -10.0,
        ],
        dtype=np.float32,
    )
    max_pattern = np.array(
        [
            0.0, 0.0, 0.0, 1.0, np.inf, -np.inf, -1.0, 1.0,
            0.0, 0.0, 100.0, -100.0, 3.0, -3.0, 20.0, -20.0,
        ],
        dtype=np.float32,
    )
    flat_src = np.resize(src_pattern, ROWS * COLS).astype(np.float32, copy=False)
    flat_max = np.resize(max_pattern, ROWS * COLS).astype(np.float32, copy=False)
    v1 = flat_src.reshape(ROWS, COLS)
    v2 = flat_max.reshape(ROWS, COLS)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v3 = np.exp(flat_src - flat_max).astype(np.float32, copy=False).reshape(ROWS, COLS)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    v3.reshape(-1).tofile(output_dir / "v3.bin")
    golden_v3.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
