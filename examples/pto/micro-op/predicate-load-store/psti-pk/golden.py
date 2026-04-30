#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/predicate-load-store/psti-pk
# family: predicate-load-store
# target_ops: pto.psti
# scenarios: packed-store, immediate-offset, representative-logical-elements

import argparse
from pathlib import Path

import numpy as np


SEED = 19
OUTPUT_WORDS = 8
ACTIVE_BITS = 145
PK_STORAGE_BYTES = 16


def prefix_bits(active_bits: int) -> np.ndarray:
    bits = np.zeros((256,), dtype=np.uint8)
    bits[:active_bits] = 1
    return bits


def generate(output_dir: Path, seed: int) -> None:
    del seed
    bits = prefix_bits(ACTIVE_BITS)
    packed_pk = np.packbits(bits[::2], bitorder="little")
    out = np.zeros((OUTPUT_WORDS * 4,), dtype=np.uint8)
    out[:PK_STORAGE_BYTES] = packed_pk[:PK_STORAGE_BYTES]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.zeros((OUTPUT_WORDS,), dtype=np.uint32).tofile(output_dir / "v1.bin")
    out.view(np.uint32).tofile(output_dir / "golden_v1.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate packed predicate golden for VPTO micro-op psti-pk validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
