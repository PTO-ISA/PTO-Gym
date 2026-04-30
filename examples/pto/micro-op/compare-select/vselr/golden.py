#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/compare-select/vselr
# family: compare-select
# target_ops: pto.vselr
# scenarios: core-f32, full-mask, explicit-lane-index
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-3.0, 3.0, size=(ROWS, COLS)).astype(np.float32, copy=False)
    src = v1.reshape(16, 64)
    lane_ids = np.arange(64, dtype=np.int32)
    idx = np.empty((16, 64), dtype=np.int32)
    for row in range(16):
        idx[row] = (lane_ids[::-1] + row * 3 + (lane_ids // 8) * 3) % 64
    golden_v3 = np.take_along_axis(src, idx, axis=1).astype(np.float32, copy=False).reshape(ROWS, COLS)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    idx.reshape(-1).tofile(output_dir / "v2.bin")
    v3.reshape(-1).tofile(output_dir / "v3.bin")
    golden_v3.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vselr validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where v1.bin/v2.bin/v3.bin/golden_v3.bin are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Numpy random seed.",
    )
    args = parser.parse_args()

    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
