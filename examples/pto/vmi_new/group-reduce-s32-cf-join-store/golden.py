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
GROUP_SIZE = 32
INPUT_ELEMS = ROWS * GROUP_SIZE
SENTINEL = np.float32(-777.0)


def generate(output_dir: Path) -> None:
    src = np.empty(INPUT_ELEMS, dtype=np.float32)
    dst = np.full(ROWS, SENTINEL, dtype=np.float32)
    golden = np.empty(ROWS, dtype=np.float32)

    base_row = np.linspace(-0.5, 0.75, GROUP_SIZE, dtype=np.float32)
    for row in range(ROWS):
        begin = row * GROUP_SIZE
        values = base_row + np.float32(row) * np.float32(0.03125)
        src[begin : begin + GROUP_SIZE] = values
        golden[row] = np.sum(values, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    dst.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
