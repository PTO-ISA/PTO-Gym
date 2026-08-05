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

ROWS = 4
INPUT_COLS = 8
OUT_COLS = 4
SCALE_BYTES = 4
SENTINEL_U8 = np.uint8(0xA5)

Q_VALUES = np.array([1.0, -1.0, 0.5, 57344.0], dtype=np.float32)
F8E5M2_BYTES = np.array([0x3C, 0xBC, 0x38, 0x7B], dtype=np.uint8)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def generate(output_dir: Path) -> None:
    x2 = f32_to_bf16_bits(Q_VALUES / np.float32(16.0))
    x1 = f32_to_bf16_bits(np.full(OUT_COLS, np.float32(16.0), dtype=np.float32))
    src_row = np.concatenate([x2, x1])
    src = np.tile(src_row, (ROWS, 1))
    golden_out = np.tile(F8E5M2_BYTES, (ROWS, 1)).astype(np.uint8)
    golden_scale = np.full(SCALE_BYTES, np.uint8(0x7F), dtype=np.uint8)

    out = np.full((ROWS, OUT_COLS), SENTINEL_U8, dtype=np.uint8)
    scale = np.full(SCALE_BYTES, SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    out.tofile(output_dir / "v2.bin")
    scale.tofile(output_dir / "v3.bin")
    golden_out.tofile(output_dir / "golden_v2.bin")
    golden_scale.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
