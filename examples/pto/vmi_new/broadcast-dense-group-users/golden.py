#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

ROWS = 8
COLS = 32
SCALE = np.float32(0.5)
SENTINEL = np.float32(-777.0)


def generate(output_dir: Path) -> None:
    base = np.linspace(-0.875, 0.625, COLS, dtype=np.float32)
    src = np.empty((ROWS, COLS), dtype=np.float32)
    for row in range(ROWS):
        src[row, :] = base + np.float32(row) * np.float32(0.03125)
    copy = np.full((ROWS, COLS), SENTINEL, dtype=np.float32)
    sums = np.full(ROWS, SENTINEL, dtype=np.float32)
    golden_copy = src + SCALE
    golden_sum = np.sum(src * SCALE, axis=1, dtype=np.float32).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.reshape(-1).tofile(output_dir / "v1.bin")
    copy.reshape(-1).tofile(output_dir / "v2.bin")
    sums.tofile(output_dir / "v3.bin")
    golden_copy.reshape(-1).astype(np.float32).tofile(output_dir / "golden_v2.bin")
    golden_sum.astype(np.float32).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
