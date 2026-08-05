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
COLS = 128
INT8_MAX = np.float32(127.0)
SENTINEL_F32 = np.float32(-777.0)
SENTINEL_U8 = np.uint8(0xA5)

ROW_Q = np.array(
    [
        -127,
        -96,
        -64,
        -32,
        -7,
        -1,
        0,
        1,
        7,
        16,
        31,
        63,
        95,
        120,
        127,
        64,
    ],
    dtype=np.float32,
)
COL_SCALES = np.array([0.125, 0.25, 0.5, 1.0, 2.0], dtype=np.float32)


def generate(output_dir: Path) -> None:
    q = np.tile(ROW_Q, (ROWS + len(ROW_Q) - 1) // len(ROW_Q))[:ROWS]
    col_scales = np.tile(COL_SCALES, (COLS + len(COL_SCALES) - 1) // len(COL_SCALES))[:COLS]

    src = (q[:, None] * col_scales[None, :]).astype(np.float16)
    x_f32 = src.astype(np.float32)
    golden_scale = (np.max(np.abs(x_f32), axis=0) / INT8_MAX).astype(np.float32)
    scale_safe = np.where(golden_scale > 0, golden_scale, np.ones_like(golden_scale))
    golden_out = np.round(x_f32 / scale_safe[None, :]).astype(np.float32)
    golden_out = np.clip(golden_out, -128, 127).astype(np.int8)

    scale = np.full(COLS, SENTINEL_F32, dtype=np.float32)
    out = np.full((ROWS, COLS), SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.view(np.uint16).tofile(output_dir / "v1.bin")
    scale.tofile(output_dir / "v2.bin")
    out.tofile(output_dir / "v3.bin")
    golden_scale.tofile(output_dir / "golden_v2.bin")
    golden_out.view(np.uint8).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
