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
SENTINEL = np.float32(-777.0)


def fill_matrix(cols: int, base_start: float, row_step: float) -> np.ndarray:
    base = np.linspace(base_start, base_start + 1.0, cols, dtype=np.float32)
    out = np.empty((ROWS, cols), dtype=np.float32)
    for row in range(ROWS):
        out[row, :] = base + np.float32(row) * np.float32(row_step)
    return out


def write_case(output_dir: Path, matrix: np.ndarray, src_name: str, dst_name: str, golden_name: str) -> None:
    dst = np.full(ROWS, SENTINEL, dtype=np.float32)
    golden = np.sum(matrix, axis=1, dtype=np.float32).astype(np.float32)
    matrix.reshape(-1).tofile(output_dir / src_name)
    dst.tofile(output_dir / dst_name)
    golden.tofile(output_dir / golden_name)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_case(output_dir, fill_matrix(8, -0.5, 0.03125), "v1.bin", "v4.bin", "golden_v4.bin")
    write_case(output_dir, fill_matrix(16, -0.75, 0.046875), "v2.bin", "v5.bin", "golden_v5.bin")
    write_case(output_dir, fill_matrix(32, -0.875, 0.0625), "v3.bin", "v6.bin", "golden_v6.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
