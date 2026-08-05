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

ROWS = 2
ROW_ELEMS = 256
ROW_STRIDE = 320
TOTAL_ELEMS = ROWS * ROW_STRIDE
F16_VALUES = np.array([0.125, 0.25], dtype=np.float16)
VALUES = np.array([0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 4.0, -4.0], dtype=np.float32)
F8E4M3FN_BYTES = np.array([0x00, 0x38, 0xB8, 0x30, 0x40, 0xC0, 0x48, 0xC8], dtype=np.uint8)
SENTINEL = np.float32(-123.25)


def generate(output_dir: Path) -> None:
    repeats = (ROW_ELEMS + len(VALUES) - 1) // len(VALUES)
    row_f8 = np.tile(F8E4M3FN_BYTES, repeats)[:ROW_ELEMS].astype(np.uint8)
    row_decoded_f8 = np.tile(VALUES, repeats)[:ROW_ELEMS].astype(np.float32)

    src_f16 = np.zeros(TOTAL_ELEMS, dtype=np.float16)
    src_f8 = np.zeros(TOTAL_ELEMS, dtype=np.uint8)
    dst = np.full(TOTAL_ELEMS, SENTINEL, dtype=np.float32)
    golden = np.full(TOTAL_ELEMS, SENTINEL, dtype=np.float32)

    for row in range(ROWS):
        begin = row * ROW_STRIDE
        end = begin + ROW_ELEMS
        src_f16[begin:end] = F16_VALUES[row]
        src_f8[begin:end] = np.roll(row_f8, row)
        decoded_f8 = np.roll(row_decoded_f8, row)
        reduction = np.sum(src_f16[begin:end].astype(np.float32), dtype=np.float32)
        golden[begin:end] = decoded_f8 * reduction

    output_dir.mkdir(parents=True, exist_ok=True)
    src_f16.tofile(output_dir / "v1.bin")
    src_f8.tofile(output_dir / "v2.bin")
    dst.tofile(output_dir / "v3.bin")
    golden.astype(np.float32, copy=False).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
