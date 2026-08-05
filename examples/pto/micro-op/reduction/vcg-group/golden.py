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


GROUPS = 8
ELEMS_PER_GROUP = 8
LANES = GROUPS * ELEMS_PER_GROUP


def generate(output_dir: Path) -> None:
    src = np.array(
        [
            -7.0, 1.0, 3.5, -2.0, 9.0, -4.5, 6.0, 0.5,
            8.0, -1.0, -3.0, 4.0, 2.0, -6.0, 5.5, 7.0,
            -0.0, 0.0, -5.0, 5.0, 11.0, -12.0, 13.0, -14.0,
            1.25, 2.25, 3.25, 4.25, -8.0, -9.0, 10.0, -10.0,
            15.0, 14.0, 13.0, 12.0, -1.5, -2.5, -3.5, -4.5,
            -20.0, -19.0, -18.0, -17.0, 16.0, 15.5, 14.5, 13.5,
            0.25, -0.75, 1.5, -2.25, 3.0, -3.75, 4.5, -5.25,
            31.0, -32.0, 33.0, -34.0, 35.0, -36.0, 37.0, -38.0,
        ],
        dtype=np.float32,
    )
    groups = src.reshape(GROUPS, ELEMS_PER_GROUP)

    golden_add = np.zeros(LANES, dtype=np.float32)
    golden_max = np.zeros(LANES, dtype=np.float32)
    golden_min = np.zeros(LANES, dtype=np.float32)
    golden_add[:GROUPS] = np.sum(groups, axis=1, dtype=np.float32)
    golden_max[:GROUPS] = np.max(groups, axis=1)
    golden_min[:GROUPS] = np.min(groups, axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "src.bin")
    golden_add.tofile(output_dir / "golden_add.bin")
    golden_max.tofile(output_dir / "golden_max.bin")
    golden_min.tofile(output_dir / "golden_min.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
