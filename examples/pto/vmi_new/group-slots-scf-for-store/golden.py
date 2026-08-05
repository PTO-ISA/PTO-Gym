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
COLS = 16
SENTINEL = np.float32(-777.0)


def generate(output_dir: Path) -> None:
    init = np.linspace(-0.25, 0.625, ROWS, dtype=np.float32)
    base = np.linspace(-0.75, 0.25, COLS, dtype=np.float32)
    src = np.empty((ROWS, COLS), dtype=np.float32)
    for row in range(ROWS):
        src[row, :] = base + np.float32(row) * np.float32(0.03125)
    dst = np.full(ROWS, SENTINEL, dtype=np.float32)
    golden = init + np.float32(2.0) * np.sum(src, axis=1, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    init.tofile(output_dir / "v1.bin")
    src.reshape(-1).tofile(output_dir / "v2.bin")
    dst.tofile(output_dir / "v3.bin")
    golden.astype(np.float32).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
