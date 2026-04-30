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


LANES = 256
SEED = 19
THRESHOLD = np.int8(5)
OUTPUT_BYTES = 32


def encode_b8_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
    for i, bit in enumerate(mask.astype(np.uint8, copy=False)):
        if bit:
            byte_index = i // 8
            bit_shift = i % 8
            out[byte_index] |= np.uint8(1 << bit_shift)
    return out


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-128, 127, size=(LANES,), dtype=np.int8)
    mask = np.greater(v1, THRESHOLD)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    np.zeros((OUTPUT_BYTES,), dtype=np.uint8).tofile(output_dir / "v2.bin")
    encode_b8_mask(mask).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inputs/golden for VPTO vcmps-i8-signed.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
