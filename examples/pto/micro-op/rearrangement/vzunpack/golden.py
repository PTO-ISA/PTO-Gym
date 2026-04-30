#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/rearrangement/vzunpack
# family: rearrangement
# target_ops: pto.vzunpack
# scenarios: pack-unpack, zero-extend
# NOTE: zero-extending unpack of the lower half of each 128-lane ui16 chunk.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


INPUT_ELEMS = 2048
OUTPUT_ELEMS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(0, np.iinfo(np.uint16).max + 1, size=INPUT_ELEMS, dtype=np.uint16)
    v2 = np.zeros(OUTPUT_ELEMS, dtype=np.uint32)
    golden_v2 = np.zeros(OUTPUT_ELEMS, dtype=np.uint32)
    for src_base in range(0, INPUT_ELEMS, 128):
        dst_base = (src_base // 128) * 64
        golden_v2[dst_base : dst_base + 64] = v1[src_base : src_base + 64].astype(np.uint32)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vzunpack validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
