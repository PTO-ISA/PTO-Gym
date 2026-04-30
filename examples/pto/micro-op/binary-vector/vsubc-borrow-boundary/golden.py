#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/binary-vector/vsubc-borrow-boundary
# family: binary-vector
# target_ops: pto.vsubc
# scenarios: core-u32-unsigned, full-mask, carry-chain
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


LANES = 64
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    del seed
    v1 = np.zeros(LANES, dtype=np.uint32)
    v2 = np.zeros(LANES, dtype=np.uint32)
    pattern_lhs = np.array([0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000], dtype=np.uint32)
    pattern_rhs = np.array([0x00000001, 0x00000002, 0x80000000, 0xFFFFFFFF], dtype=np.uint32)
    reps = LANES // pattern_lhs.size
    v1[:] = np.tile(pattern_lhs, reps)
    v2[:] = np.tile(pattern_rhs, reps)
    no_borrow = v1 >= v2
    result = (v1 - v2).astype(np.uint32, copy=False)
    packed = np.zeros(256, dtype=np.uint8)
    for idx, bit in enumerate(no_borrow):
        if not bit:
            continue
        byte = idx // 2
        if idx % 2 == 0:
            packed[byte] |= np.uint8(0x1)
        else:
            packed[byte] |= np.uint8(0x10)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    np.zeros(LANES, dtype=np.uint32).tofile(output_dir / "v3.bin")
    np.zeros(256, dtype=np.uint8).tofile(output_dir / "v4.bin")
    result.tofile(output_dir / "golden_v3.bin")
    packed.tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
