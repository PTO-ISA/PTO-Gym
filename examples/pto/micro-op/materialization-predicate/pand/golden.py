#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/materialization-predicate/pand
# family: materialization-predicate
# target_ops: pto.pand
# scenarios: predicate-transform, logical-and
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


SEED = 19
OUTPUT_WORDS = 32
PREFIX_BITS = 13
SUFFIX_BITS = 7
PREDICATE_BITS = 256
NIBBLE_COUNT = PREDICATE_BITS // 2


def pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    words = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)
    for idx, nibble in enumerate(nibbles):
        words[idx // 8] |= np.uint32(int(nibble) & 0xF) << np.uint32((idx % 8) * 4)
    return words


def generate(output_dir: Path, seed: int) -> None:
    del seed
    output_init = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)
    lhs = np.zeros((NIBBLE_COUNT,), dtype=np.uint8)
    rhs = np.zeros((NIBBLE_COUNT,), dtype=np.uint8)
    lhs[:PREFIX_BITS] = 1
    rhs[:SUFFIX_BITS] = 1
    golden = pack_nibbles(lhs & rhs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_init.tofile(output_dir / "v1.bin")
    golden.tofile(output_dir / "golden_v1.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate packed predicate golden for VPTO micro-op validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
