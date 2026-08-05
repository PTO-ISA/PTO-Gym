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
COLS = 64
SCALE_SLOTS = 16
INT8_MAX = np.float32(127.0)
SENTINEL_F32 = np.float32(-777.0)
SENTINEL_U8 = np.uint8(0xA5)
SMOOTH_VALUES = np.array(
    [0.5, 0.75, 1.0, 1.25, 1.5, 0.625, 0.875, 1.125],
    dtype=np.float32,
)

Q_VALUES = np.array(
    [-127, -96, -64, -32, -7, -1, 0, 1, 7, 16, 31, 63, 95, 120, 127],
    dtype=np.float32,
)
ROW_SCALES = np.array(
    [
        0.25,
        0.5,
        1.0,
        2.0,
        0.375,
        0.75,
        1.5,
        3.0,
        0.125,
        0.625,
        1.25,
        2.5,
        0.3125,
        0.9375,
        1.875,
        3.75,
    ],
    dtype=np.float32,
)


def generate(output_dir: Path) -> None:
    repeats = (COLS + len(Q_VALUES) - 1) // len(Q_VALUES)
    q_row = np.tile(Q_VALUES, repeats)[:COLS].astype(np.float32)
    smooth_repeats = (COLS + len(SMOOTH_VALUES) - 1) // len(SMOOTH_VALUES)
    smooth = np.tile(SMOOTH_VALUES, smooth_repeats)[:COLS].astype(np.float16)

    src = np.empty((ROWS, COLS), dtype=np.float16)
    golden_scale = np.full(SCALE_SLOTS, SENTINEL_F32, dtype=np.float32)
    golden_out = np.empty((ROWS, COLS), dtype=np.int8)
    for row in range(ROWS):
        src[row] = (q_row * ROW_SCALES[row]).astype(np.float16)
        x_f32 = src[row].astype(np.float32) * smooth.astype(np.float32)
        scale = (np.max(np.abs(x_f32)) / INT8_MAX).astype(np.float32)
        golden_scale[(row // 4) * 8 + (row % 4)] = scale
        quant = np.round(x_f32 / scale).astype(np.float32)
        golden_out[row] = np.clip(quant, -128, 127).astype(np.int8)

    scale = np.full(SCALE_SLOTS, SENTINEL_F32, dtype=np.float32)
    out = np.full((ROWS, COLS), SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.view(np.uint16).tofile(output_dir / "v1.bin")
    smooth.view(np.uint16).tofile(output_dir / "v2.bin")
    scale.tofile(output_dir / "v3.bin")
    out.tofile(output_dir / "v4.bin")
    golden_scale.tofile(output_dir / "golden_v3.bin")
    golden_out.view(np.uint8).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
