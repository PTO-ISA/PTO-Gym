#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/binary-vector/vsubc
# family: binary-vector
# target_ops: pto.vsubc
# scenarios: core-u32-unsigned, full-mask, carry-chain

import argparse
from pathlib import Path

import numpy as np


LANES = 64
SEED = 19


def pack_mask_nibbles(bits):
    out = np.zeros(256, dtype=np.uint8)
    for idx, bit in enumerate(bits):
        if not bit:
            continue
        byte = idx // 2
        if idx % 2 == 0:
            out[byte] |= np.uint8(0x1)
        else:
            out[byte] |= np.uint8(0x10)
    return out


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(0, 0xFFFFFFFF, size=LANES, dtype=np.uint32)
    v2 = rng.integers(0, 0xFFFFFFFF, size=LANES, dtype=np.uint32)
    diff = (v1 - v2).astype(np.uint32, copy=False)
    no_borrow = v1 >= v2

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    np.zeros(LANES, dtype=np.uint32).tofile(output_dir / "v3.bin")
    np.zeros(256, dtype=np.uint8).tofile(output_dir / "v4.bin")
    diff.tofile(output_dir / "golden_v3.bin")
    pack_mask_nibbles(no_borrow).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
