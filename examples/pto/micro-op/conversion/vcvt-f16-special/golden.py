#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/conversion/vcvt-f16-special
# family: conversion
# target_ops: pto.vcvt
# scenarios: f16-to-f32, exceptional-values
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    del seed
    special = np.array(
        [
            np.float16(0.0),
            np.float16(-0.0),
            np.float16(1.0),
            np.float16(-1.0),
            np.float16(np.inf),
            np.float16(-np.inf),
            np.float16(np.nan),
            np.float16(65504.0),
            np.float16(-65504.0),
            np.float16(6.1035e-05),
            np.float16(-6.1035e-05),
            np.float16(5.9605e-08),
            np.float16(-5.9605e-08),
            np.float16(123.75),
            np.float16(-123.75),
            np.float16(0.33325),
        ],
        dtype=np.float16,
    )
    v1 = np.resize(special, ROWS * COLS).reshape(ROWS, COLS)
    v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v2 = v1.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    golden_v2.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vcvt-f16-special validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
