#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


SEED = 19
VALUES = np.full(64, -1, dtype=np.int64)
VALUES[0] = 0
VALUES[1] = 0
VALUES[2] = 2
VALUES[3] = 1
VALUES[32] = 1
VALUES[33] = 0
VALUES[34] = 2
VALUES[35] = 1


def generate(output_dir: Path, seed: int) -> None:
    del seed
    v1 = np.full(VALUES.shape, -1, dtype=np.int64)
    golden_v1 = VALUES.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO runtime query validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
