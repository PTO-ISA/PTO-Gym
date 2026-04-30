#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/materialization-predicate/pintlv_b8-nontrivial
# family: materialization-predicate
# target_ops: pto.pintlv_b8
# scenarios: predicate-transform, lane-order, representative-logical-elements
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


SEED = 19
OUTPUT_WORDS = 32
GOLDEN_PREFIX_WORDS = np.array([33707863, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018, 33686018], dtype=np.uint32)


def generate(output_dir: Path, seed: int) -> None:
    del seed
    output_init = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)
    golden = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)
    golden[: GOLDEN_PREFIX_WORDS.size] = GOLDEN_PREFIX_WORDS

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
