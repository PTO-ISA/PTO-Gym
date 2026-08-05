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

ELEMS = 1024
LOGICAL_ELEMS = 1000
SEED = 29
SCALE = np.float32(0.5)
SENTINEL = np.float16(-17.5)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    src = rng.uniform(-8.0, 8.0, size=ELEMS).astype(np.float32)
    dst = np.full(ELEMS, SENTINEL, dtype=np.float16)
    golden = np.full(ELEMS, SENTINEL, dtype=np.float16)
    golden[:LOGICAL_ELEMS] = (src[:LOGICAL_ELEMS] * SCALE).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    dst.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
