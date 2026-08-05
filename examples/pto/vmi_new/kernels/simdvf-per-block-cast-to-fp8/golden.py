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

ROWS = 16
COLS = 256
SCALE_ROWS = 4
SCALE_COLS = 8
TOKENS_PER_SCALE_ROW = 4
CHANNELS_PER_SCALE = 32
SENTINEL_U8 = np.uint8(0xA5)

Q_VALUES = np.array(
    [0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 4.0, -4.0, 448.0], dtype=np.float32
)
F8E4M3FN_BYTES = np.array(
    [0x00, 0x38, 0xB8, 0x30, 0x40, 0xC0, 0x48, 0xC8, 0x7E], dtype=np.uint8
)


def generate(output_dir: Path) -> None:
    scale = np.array(
        [
            [0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0],
            [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
            [1.0, 2.0, 4.0, 0.25, 1.0, 2.0, 4.0, 0.25],
            [2.0, 4.0, 0.25, 0.5, 2.0, 4.0, 0.25, 0.5],
        ],
        dtype=np.float32,
    )

    repeats = (CHANNELS_PER_SCALE + len(Q_VALUES) - 1) // len(Q_VALUES)
    q_block = np.tile(Q_VALUES, repeats)[:CHANNELS_PER_SCALE].astype(np.float32)
    f8_block = np.tile(F8E4M3FN_BYTES, repeats)[:CHANNELS_PER_SCALE]

    src = np.empty((ROWS, COLS), dtype=np.float16)
    golden_out = np.empty((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        scale_row = row // TOKENS_PER_SCALE_ROW
        for scale_col in range(SCALE_COLS):
            start = scale_col * CHANNELS_PER_SCALE
            stop = start + CHANNELS_PER_SCALE
            src[row, start:stop] = (q_block / scale[scale_row, scale_col]).astype(
                np.float16
            )
            golden_out[row, start:stop] = f8_block

    out = np.full((ROWS, COLS), SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.view(np.uint16).tofile(output_dir / "v1.bin")
    scale.reshape(-1).tofile(output_dir / "v2.bin")
    out.tofile(output_dir / "v3.bin")
    golden_out.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
