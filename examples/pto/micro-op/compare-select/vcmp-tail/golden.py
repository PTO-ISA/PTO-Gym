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


LANES = 64
LOGICAL_ELEMS = 53
SEED = 19
OUTPUT_BYTES = 32


def encode_b32_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
    for i, bit in enumerate(mask.astype(np.uint8, copy=False)):
        if bit:
            byte_index = i // 2
            nibble_shift = 4 * (i % 2)
            out[byte_index] |= np.uint8(1 << nibble_shift)
    return out


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-6.0, 6.0, size=(LANES,)).astype(np.float32)
    delta = rng.uniform(0.1, 2.0, size=(LANES,)).astype(np.float32)
    mode = np.arange(LANES, dtype=np.int32) % 5

    v2 = np.empty((LANES,), dtype=np.float32)
    v2[mode == 0] = v1[mode == 0] + delta[mode == 0]
    v2[mode == 1] = v1[mode == 1] - delta[mode == 1]
    v2[mode == 2] = v1[mode == 2]
    v2[mode == 3] = np.nextafter(v1[mode == 3], np.float32(np.inf))
    v2[mode == 4] = np.nextafter(v1[mode == 4], np.float32(-np.inf))

    v1[:10] = np.array([-3.0, -1.0, -0.0, 0.0, 0.25, 1.0, 2.0, 4.0, -4.0, 6.0], dtype=np.float32)
    v2[:10] = np.array([
        -2.0,
        -2.0,
        0.0,
        np.nextafter(np.float32(0.0), np.float32(np.inf)),
        0.25,
        np.nextafter(np.float32(1.0), np.float32(-np.inf)),
        3.0,
        3.0,
        np.nextafter(np.float32(-4.0), np.float32(np.inf)),
        6.0,
    ], dtype=np.float32)

    mask = np.less(v1, v2)
    mask[LOGICAL_ELEMS:] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    np.zeros((OUTPUT_BYTES,), dtype=np.uint8).tofile(output_dir / "v3.bin")
    encode_b32_mask(mask).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inputs/golden for VPTO vcmp-tail.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
