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

ELEMS = 128
ACTIVE = 96
SEED = 29
SENTINEL32 = np.float32(-901.25)
SENTINEL16 = np.float16(-17.5)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    src = rng.uniform(-8.0, 8.0, size=ELEMS).astype(np.float32)
    out32 = np.full(ELEMS, SENTINEL32, dtype=np.float32)
    out16 = np.full(ELEMS, SENTINEL16, dtype=np.float16)
    golden32 = out32.copy()
    golden16 = out16.copy()
    golden32[:ACTIVE] = src[:ACTIVE]
    golden16[:ACTIVE] = src[:ACTIVE].astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    out32.tofile(output_dir / "v2.bin")
    out16.tofile(output_dir / "v3.bin")
    golden32.tofile(output_dir / "golden_v2.bin")
    golden16.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
