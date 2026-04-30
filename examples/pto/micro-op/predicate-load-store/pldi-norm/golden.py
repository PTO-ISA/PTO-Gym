#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/predicate-load-store/pldi-norm
# family: predicate-load-store
# target_ops: pto.pldi
# scenarios: packed-load, immediate-offset, representative-logical-elements

import argparse
from pathlib import Path

import numpy as np


SEED = 19
ACTIVE_BITS = 145
OUTPUT_BYTES = 1024
VECTOR_BYTES = 256
PACKED_BYTES = 32


def prefix_bits(active_bits: int) -> np.ndarray:
    bits = np.zeros((256,), dtype=np.uint8)
    bits[:active_bits] = 1
    return bits


def make_input_buffer(bits: np.ndarray) -> np.ndarray:
    packed = np.packbits(bits.astype(np.uint8, copy=False), bitorder="little")
    ones = np.ones((VECTOR_BYTES,), dtype=np.uint8)
    zeros = np.zeros((VECTOR_BYTES,), dtype=np.uint8)
    out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
    out[:PACKED_BYTES] = packed[:PACKED_BYTES]
    out[PACKED_BYTES : PACKED_BYTES + VECTOR_BYTES] = ones
    out[PACKED_BYTES + VECTOR_BYTES : PACKED_BYTES + 2 * VECTOR_BYTES] = zeros
    return out


def expected_selected_bytes(bits: np.ndarray) -> np.ndarray:
    out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
    out[:VECTOR_BYTES] = bits.astype(np.uint8, copy=False)
    return out


def generate(output_dir: Path, seed: int) -> None:
    del seed
    bits = prefix_bits(ACTIVE_BITS)
    input_buffer = make_input_buffer(bits)
    golden = expected_selected_bytes(bits)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_buffer.tofile(output_dir / "v1.bin")
    np.zeros((OUTPUT_BYTES,), dtype=np.uint8).tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw packed predicate input/golden for VPTO micro-op pldi-norm validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
