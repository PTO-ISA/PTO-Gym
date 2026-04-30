#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/conversion/vcvt-u32-to-u8-part-p0123
# family: conversion
# target_ops: pto.vcvt
# scenarios: u32-to-u8, sat, part-p0123

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 23
CHUNK = 256
SUBCHUNK = 64
U8_MAX = np.iinfo(np.uint8).max


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2000, size=ELEMS, dtype=np.uint32)
    edge = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            7,
            15,
            31,
            63,
            127,
            128,
            129,
            254,
            255,
            256,
            257,
            511,
            512,
            1023,
            65535,
            0xFFFFFFFF,
        ],
        dtype=np.uint32,
    )
    data[: edge.size] = edge

    clipped = np.clip(data, 0, U8_MAX).astype(np.uint8)
    golden = np.empty(ELEMS, dtype=np.uint8)
    for offset in range(0, ELEMS, CHUNK):
        p0 = clipped[offset : offset + SUBCHUNK]
        p1 = clipped[offset + SUBCHUNK : offset + 2 * SUBCHUNK]
        p2 = clipped[offset + 2 * SUBCHUNK : offset + 3 * SUBCHUNK]
        p3 = clipped[offset + 3 * SUBCHUNK : offset + 4 * SUBCHUNK]
        merged = np.empty(CHUNK, dtype=np.uint8)
        merged[0::4] = p0
        merged[1::4] = p1
        merged[2::4] = p2
        merged[3::4] = p3
        golden[offset : offset + CHUNK] = merged

    output_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.uint8).tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
