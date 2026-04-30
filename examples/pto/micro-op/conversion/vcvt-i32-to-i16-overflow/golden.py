#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/conversion/vcvt-i32-to-i16-overflow
# family: conversion
# target_ops: pto.vcvt
# scenarios: i32-to-i16, integer-overflow

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19
I16_MIN = np.iinfo(np.int16).min
I16_MAX = np.iinfo(np.int16).max


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.integers(-200000, 200000, size=ELEMS, dtype=np.int32)
    edge = np.array([
        -40000, -32769, -32768, -32767, -1, 0, 1, 32766,
        32767, 32768, 40000, 70000, -70000, 65535, -65535, 123456,
    ], dtype=np.int32)
    data[:edge.size] = edge
    clipped = np.clip(data, I16_MIN, I16_MAX).astype(np.int16)
    golden = np.zeros(ELEMS, dtype=np.int16)
    for offset in range(0, ELEMS, 128):
        lower = clipped[offset : offset + 64]
        upper = clipped[offset + 64 : offset + 128]
        merged = np.empty(128, dtype=np.int16)
        merged[0::2] = lower
        merged[1::2] = upper
        golden[offset : offset + 128] = merged

    output_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.int16).tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
