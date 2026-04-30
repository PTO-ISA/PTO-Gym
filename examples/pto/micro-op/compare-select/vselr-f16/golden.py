#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/compare-select/vselr-f16
# family: compare-select
# target_ops: pto.vselr
# scenarios: core-f16, full-mask, explicit-lane-index

import argparse
from pathlib import Path

import numpy as np


ROWS = 8
COLS = 128
SEED = 23


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    src = rng.uniform(-6.0, 6.0, size=(ROWS, COLS)).astype(np.float16, copy=False)
    lane_ids = np.arange(COLS, dtype=np.uint16)
    idx = np.empty((ROWS, COLS), dtype=np.uint16)
    for row in range(ROWS):
        idx[row] = (lane_ids[::-1] + row * 11 + (lane_ids % 7) * 3) % COLS
    golden = np.take_along_axis(src, idx.astype(np.int64, copy=False), axis=1).astype(np.float16, copy=False)
    out = np.zeros((ROWS, COLS), dtype=np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.view(np.uint16).reshape(-1).tofile(output_dir / "v1.bin")
    idx.reshape(-1).tofile(output_dir / "v2.bin")
    out.view(np.uint16).reshape(-1).tofile(output_dir / "v3.bin")
    golden.view(np.uint16).reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inputs/golden for vselr-f16.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
