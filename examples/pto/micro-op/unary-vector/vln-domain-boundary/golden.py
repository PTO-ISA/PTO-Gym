#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/unary-vector/vln-domain-boundary
# family: unary-vector
# target_ops: pto.vln
# scenarios: core-f32, domain-positive, exceptional-values
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(0.125, 8.0, size=(ROWS, COLS)).astype(np.float32)
    flat = v1.reshape(-1)
    flat[:8] = np.array(
        [
            np.float32(np.finfo(np.float32).tiny),
            np.float32(np.finfo(np.float32).tiny * 2.0),
            np.float32(1.0),
            np.float32(2.0),
            np.float32(16.0),
            np.float32(1024.0),
            np.float32(np.finfo(np.float32).max),
            np.float32(0.5),
        ],
        dtype=np.float32,
    )
    v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v2 = np.log(v1).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    golden_v2.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vln domain-boundary validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
