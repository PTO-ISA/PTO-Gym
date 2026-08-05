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

ELEMS = 64
ACTIVE = 48
SEED = 29
SENTINEL = np.float32(-901.25)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    src = rng.uniform(-8.0, 8.0, size=ELEMS).astype(np.float32)
    rhs = rng.uniform(-4.0, 4.0, size=ELEMS).astype(np.float32)
    dense = np.full(ELEMS, SENTINEL, dtype=np.float32)
    masked = np.full(ELEMS, SENTINEL, dtype=np.float32)
    summed = (src + rhs).astype(np.float32)
    golden_dense = src.copy()
    golden_dense[:ACTIVE] = summed[:ACTIVE]
    golden_masked = masked.copy()
    golden_masked[:ACTIVE] = summed[:ACTIVE]

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    rhs.tofile(output_dir / "v2.bin")
    dense.tofile(output_dir / "v3.bin")
    masked.tofile(output_dir / "v4.bin")
    golden_dense.tofile(output_dir / "golden_v3.bin")
    golden_masked.tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
