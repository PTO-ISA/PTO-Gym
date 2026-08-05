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
S16 = 16
S32 = 32
SENTINEL = np.float32(-777.0)


def fill_matrix(rows: int, cols: int, base_start: float, row_step: float) -> np.ndarray:
    base = np.linspace(base_start, base_start + 1.0, cols, dtype=np.float32)
    out = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        out[row, :] = base + np.float32(row) * np.float32(row_step)
    return out


def generate(output_dir: Path) -> None:
    src16 = fill_matrix(ROWS, S16, -0.75, 0.03125)
    src32 = fill_matrix(ROWS, S32, -0.875, 0.0625)
    rhs = np.linspace(-0.25, 0.625, ROWS, dtype=np.float32)
    dst16 = np.full(ROWS, SENTINEL, dtype=np.float32)
    dst32 = np.full(ROWS, SENTINEL, dtype=np.float32)

    golden16 = np.sum(src16, axis=1, dtype=np.float32).astype(np.float32) + rhs
    golden32 = np.sum(src32, axis=1, dtype=np.float32).astype(np.float32) + rhs

    output_dir.mkdir(parents=True, exist_ok=True)
    src16.reshape(-1).tofile(output_dir / "v1.bin")
    src32.reshape(-1).tofile(output_dir / "v2.bin")
    rhs.tofile(output_dir / "v3.bin")
    dst16.tofile(output_dir / "v4.bin")
    dst32.tofile(output_dir / "v5.bin")
    golden16.astype(np.float32).tofile(output_dir / "golden_v4.bin")
    golden32.astype(np.float32).tofile(output_dir / "golden_v5.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
