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

LANES = 32
FIELDS = 32
ELEMS = LANES * FIELDS


def i32_bits(value: int) -> np.int32:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return np.int32(value)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(ELEMS, -1, dtype=np.int32)
    golden_v1 = np.zeros(ELEMS, dtype=np.int32)
    for lane in range(LANES):
        base = lane * FIELDS
        golden_v1[base + 0] = lane
        golden_v1[base + 1] = lane
        golden_v1[base + 2] = 32
        golden_v1[base + 3] = 1
        golden_v1[base + 4] = 1
        golden_v1[base + 5] = 0
        golden_v1[base + 6] = 0
        golden_v1[base + 7] = 0
        golden_v1[base + 8] = 0
        golden_v1[base + 9] = i32_bits(1 << lane)
        golden_v1[base + 10] = i32_bits((1 << (lane + 1)) - 1) if lane < 31 else np.int32(-1)
        golden_v1[base + 11] = i32_bits((1 << lane) - 1)
        golden_v1[base + 12] = i32_bits(0xFFFFFFFF << lane)
        golden_v1[base + 13] = i32_bits(0xFFFFFFFF << (lane + 1)) if lane < 31 else np.int32(0)
        golden_v1[base + 14] = 0
        golden_v1[base + 15] = 1
        golden_v1[base + 16] = 0
        golden_v1[base + 17] = i32_bits(0x55555555)
        golden_v1[base + 18] = 103
        golden_v1[base + 19] = 100 + (lane - 2 if lane >= 2 else lane)
        golden_v1[base + 20] = 100 + (lane + 2 if lane <= 29 else lane)
        golden_v1[base + 21] = 100 + (lane ^ 1)
        golden_v1[base + 22] = 528
        golden_v1[base + 23] = 32
        golden_v1[base + 24] = 1
        golden_v1[base + 25] = 32
    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
