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
VALUES = np.array([0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 4.0, -4.0], dtype=np.float32)
F8E4M3FN_BYTES = np.array([0x00, 0x38, 0xB8, 0x30, 0x40, 0xC0, 0x48, 0xC8], dtype=np.uint8)
SENTINEL_F32 = np.float32(-777.0)
SENTINEL_U8 = np.uint8(0xA5)


def generate(output_dir: Path) -> None:
    src = np.empty((ROWS, GROUP_SIZE), dtype=np.float32)
    golden_out8 = np.empty((ROWS, GROUP_SIZE), dtype=np.uint8)
    for row in range(ROWS):
        value_idx = row % len(VALUES)
        if row == 0:
            src[row, :] = np.tile(VALUES, GROUP_SIZE // len(VALUES))
            golden_out8[row, :] = np.tile(F8E4M3FN_BYTES, GROUP_SIZE // len(F8E4M3FN_BYTES))
        else:
            src[row, :] = VALUES[value_idx]
            golden_out8[row, :] = F8E4M3FN_BYTES[value_idx]

    golden_sum = np.sum(src, axis=1, dtype=np.float32)
    sum_out = np.full(ROWS, SENTINEL_F32, dtype=np.float32)
    out8 = np.full(ROWS * GROUP_SIZE, SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.reshape(-1).tofile(output_dir / "v1.bin")
    sum_out.tofile(output_dir / "v2.bin")
    out8.tofile(output_dir / "v3.bin")
    golden_sum.astype(np.float32).tofile(output_dir / "golden_v2.bin")
    golden_out8.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
