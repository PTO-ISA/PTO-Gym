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
FIELDS = 20
ELEMS = 1024


def i32_bits(value: int) -> np.int32:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return np.int32(value)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.zeros(ELEMS, dtype=np.int32)
    golden_v1 = np.zeros(ELEMS, dtype=np.int32)
    for lane in range(LANES):
        base = lane * FIELDS
        group = 0 if lane < 16 else 16
        local = lane - group
        golden_v1[base + 0] = 528
        golden_v1[base + 1] = np.int32(-1)
        golden_v1[base + 2] = 0
        golden_v1[base + 3] = 528
        golden_v1[base + 4] = 32
        golden_v1[base + 5] = 1
        golden_v1[base + 6] = 528
        golden_v1[base + 7] = 32
        golden_v1[base + 8] = 1
        golden_v1[base + 9] = 100 + group + 3
        golden_v1[base + 10] = 100 + (lane - 2 if local >= 2 else lane)
        golden_v1[base + 11] = 100 + (lane + 2 if local <= 13 else lane)
        golden_v1[base + 12] = (lane ^ 1) + 1
        golden_v1[base + 13] = group + 4
        golden_v1[base + 14] = 1
        golden_v1[base + 15] = 2
    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
