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

LANES = 2
LANE_STRIDE = 8
ELEMS = LANES * LANE_STRIDE


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(ELEMS, -1, dtype=np.int32)
    golden_v1 = np.full(ELEMS, -1, dtype=np.int32)
    for lane in range(LANES):
        base = lane * LANE_STRIDE
        inputs = np.array([0x10203040 + lane, -1234567 - lane], dtype=np.int32)
        v1[base : base + 2] = inputs
        golden_v1[base : base + 2] = inputs
        golden_v1[base + 2] = inputs[0]
        golden_v1[base + 3] = inputs[1]
        golden_v1[base + 4] = np.int32(inputs[0] + inputs[1])
    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
