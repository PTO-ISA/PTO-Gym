#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/binary-vector/vadd-i16-signed-overflow
# family: binary-vector
# target_ops: pto.vadd
# scenarios: core-i16-signed, full-mask, integer-overflow

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19


def wrap_add_i16(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    bits = lhs.view(np.uint16).astype(np.uint32) + rhs.view(np.uint16).astype(np.uint32)
    return (bits & 0xFFFF).astype(np.uint16).view(np.int16)


def generate(output_dir: Path, seed: int) -> None:
    del seed
    lhs_pattern = np.array(
        [32767, 32760, -32768, -32760, 1000, -1000, 12345, -12345],
        dtype=np.int16,
    )
    rhs_pattern = np.array(
        [1, 100, -1, -100, 30000, -30000, 23456, -23456],
        dtype=np.int16,
    )
    repeats = ELEMS // lhs_pattern.size
    v1 = np.tile(lhs_pattern, repeats)
    v2 = np.tile(rhs_pattern, repeats)
    v3 = np.zeros(ELEMS, dtype=np.int16)
    golden_v3 = wrap_add_i16(v1, v2)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    v3.tofile(output_dir / "v3.bin")
    golden_v3.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
