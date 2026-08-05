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

ROWS = 6
GROUP_SIZE = 64
OUTPUT_STRIDE = 8
SENTINEL = np.float32(-777.0)


def generate(output_dir: Path) -> None:
    src = np.empty((ROWS, GROUP_SIZE), dtype=np.float32)
    base = np.linspace(-0.625, 0.875, GROUP_SIZE, dtype=np.float32)
    for row in range(ROWS):
        src[row, :] = base + np.float32(row) * np.float32(0.046875)

    dst = np.full(ROWS * OUTPUT_STRIDE, SENTINEL, dtype=np.float32)
    golden = np.full(ROWS * OUTPUT_STRIDE, SENTINEL, dtype=np.float32)
    for row in range(ROWS):
        golden[row * OUTPUT_STRIDE] = np.sum(src[row, :], dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.reshape(-1).tofile(output_dir / "v1.bin")
    dst.tofile(output_dir / "v2.bin")
    golden.astype(np.float32).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
