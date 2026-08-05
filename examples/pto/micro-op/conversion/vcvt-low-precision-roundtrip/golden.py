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

BYTES = 1024


def tiled(values):
    return np.resize(np.array(values, dtype=np.uint8), BYTES)


def expected_part_roundtrip(data):
    expected = np.zeros(BYTES, dtype=np.uint8)
    for part in range(4):
        start = part * 256
        stop = start + 256
        expected[start + part : stop : 4] = data[start + part : stop : 4]
    return expected


def generate(output_dir: Path) -> None:
    inputs = {
        "f8e4": tiled(
            [
                0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40,
                0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0xB0, 0xB8,
                0xC0, 0xC8, 0xD0, 0xD8, 0xE0, 0xE8, 0xF0,
            ]
        ),
        "f8e5": tiled(
            [
                0x04, 0x08, 0x10, 0x20, 0x30, 0x34, 0x38, 0x3C,
                0x40, 0x44, 0x48, 0x4C, 0x50, 0x60, 0x70, 0xB8,
                0xBC, 0xC0, 0xC4, 0xC8, 0xD0, 0xE0, 0xF0,
            ]
        ),
        "hif8": tiled(
            [
                0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x48,
                0x90, 0x98, 0xA0, 0xA8, 0xB0, 0xB8, 0xC0, 0xC8,
            ]
        ),
        "f4e1": tiled([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE]),
        "f4e2": tiled([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE]),
    }
    file_pairs = (
        ("v1_f8e4_in.bin", "v6_f8e4_out.bin", "golden_v6_f8e4_out.bin", inputs["f8e4"]),
        ("v2_f8e5_in.bin", "v7_f8e5_out.bin", "golden_v7_f8e5_out.bin", inputs["f8e5"]),
        ("v3_hif8_in.bin", "v8_hif8_out.bin", "golden_v8_hif8_out.bin", inputs["hif8"]),
        ("v4_f4e1_in.bin", "v9_f4e1_out.bin", "golden_v9_f4e1_out.bin", inputs["f4e1"]),
        ("v5_f4e2_in.bin", "v10_f4e2_out.bin", "golden_v10_f4e2_out.bin", inputs["f4e2"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for input_name, output_name, golden_name, data in file_pairs:
        data.tofile(output_dir / input_name)
        np.full(BYTES, 0xA5, dtype=np.uint8).tofile(output_dir / output_name)
        expected_part_roundtrip(data).tofile(output_dir / golden_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
