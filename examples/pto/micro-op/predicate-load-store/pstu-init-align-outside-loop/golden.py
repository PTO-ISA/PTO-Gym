#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/predicate-load-store/pstu-init-align-outside-loop
# family: predicate-load-store
# target_ops: pto.pstu
# scenarios: unaligned-predicate-store, state-update, representative-logical-elements, init-align-outside-loop
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
PACKED_BYTES_PER_STORE = 8
OUTPUT_WORDS = 8


def _pack_mask_b32(active_lanes: int) -> np.ndarray:
    if active_lanes < 0 or active_lanes > 64:
        raise ValueError(f"active_lanes must be in [0, 64], got {active_lanes}")
    logical = np.zeros((64,), dtype=np.uint8)
    logical[:active_lanes] = 1
    packed = np.packbits(logical, bitorder="little")
    out = np.zeros((PACKED_BYTES_PER_STORE,), dtype=np.uint8)
    out[: packed.size] = packed
    return out


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-3.0, 3.0, size=(ROWS, COLS)).astype(np.float32)
    v2 = rng.uniform(-1.0, 1.0, size=(ROWS, COLS)).astype(np.float32)

    first = _pack_mask_b32(13)
    second = _pack_mask_b32(7)
    packed = np.zeros((OUTPUT_WORDS * np.dtype(np.uint32).itemsize,), dtype=np.uint8)
    packed[:PACKED_BYTES_PER_STORE] = first
    packed[PACKED_BYTES_PER_STORE : 2 * PACKED_BYTES_PER_STORE] = second
    packed[2 * PACKED_BYTES_PER_STORE : 3 * PACKED_BYTES_PER_STORE] = first
    packed[3 * PACKED_BYTES_PER_STORE : 4 * PACKED_BYTES_PER_STORE] = second
    output_init = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    output_init.tofile(output_dir / "v3.bin")
    packed.view(np.uint32).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op pstu-init-align-outside-loop validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where v1.bin/v2.bin/v3.bin/golden_v3.bin are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Numpy random seed.",
    )
    args = parser.parse_args()

    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
