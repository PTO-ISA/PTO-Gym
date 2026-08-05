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

ROWS = 32
COLS = 32
E8M0_BYTES = 256
SENTINEL_U8 = np.uint8(0xA5)

Q_VALUES = np.array(
    [0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 4.0, -4.0, 448.0], dtype=np.float32
)
F8E4M3FN_BYTES = np.array(
    [0x00, 0x38, 0xB8, 0x30, 0x40, 0xC0, 0x48, 0xC8, 0x7E], dtype=np.uint8
)


def generate(output_dir: Path) -> None:
    repeats = (COLS + len(Q_VALUES) - 1) // len(Q_VALUES)
    q_row = np.tile(Q_VALUES, repeats)[:COLS].astype(np.float32)
    f8_row = np.tile(F8E4M3FN_BYTES, repeats)[:COLS].astype(np.uint8)

    src = np.tile(q_row / np.float32(256.0), (ROWS, 1)).astype(np.float32)
    golden_fp8 = np.tile(f8_row, (ROWS, 1)).astype(np.uint8)
    golden_e8m0 = np.full(E8M0_BYTES, SENTINEL_U8, dtype=np.uint8)
    golden_e8m0[:ROWS] = np.uint8(0x77)

    out_fp8 = np.full((ROWS, COLS), SENTINEL_U8, dtype=np.uint8)
    out_e8m0 = np.full(E8M0_BYTES, SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    out_fp8.tofile(output_dir / "v2.bin")
    out_e8m0.tofile(output_dir / "v3.bin")
    golden_fp8.tofile(output_dir / "golden_v2.bin")
    golden_e8m0.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
