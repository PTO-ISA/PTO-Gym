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


SIGNED_LANES = 64
UNSIGNED_LANES = 128
OUTPUT_ELEMS = SIGNED_LANES + UNSIGNED_LANES


def signed_pattern(size: int, salt: int) -> np.ndarray:
    lanes = np.arange(size, dtype=np.int64)
    magnitude = ((lanes + 1) * (104729 + salt * 8191)) % 2_000_000_000 + 1
    signs = np.where(((lanes + salt) & 1) == 0, 1, -1)
    return (magnitude * signs).astype(np.int32)


def unsigned_pattern(size: int, salt: int) -> np.ndarray:
    lanes = np.arange(size, dtype=np.uint64)
    values = np.uint64(0x80000001 + salt) + lanes * np.uint64(0x0010003D)
    return (values & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def product_halves(lhs: np.ndarray, rhs: np.ndarray, *, signed: bool) -> tuple[np.ndarray, np.ndarray]:
    if signed:
        product = lhs.astype(np.int64) * rhs.astype(np.int64)
        low = (product & np.int64(0xFFFFFFFF)).astype(np.uint32)
        high = ((product >> np.int64(32)) & np.int64(0xFFFFFFFF)).astype(np.uint32)
    else:
        product = lhs.astype(np.uint64) * rhs.astype(np.uint64)
        low = (product & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        high = ((product >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return low, high


def generate(output_dir: Path) -> None:
    signed_lhs = signed_pattern(SIGNED_LANES, 1)
    signed_rhs = signed_pattern(SIGNED_LANES, 2)
    signed_lhs[0], signed_rhs[0] = np.int32(-1), np.int32(2)
    signed_lhs[63], signed_rhs[63] = np.int32(-0x80000000), np.int32(-1)

    unsigned_lhs = unsigned_pattern(UNSIGNED_LANES, 3)
    unsigned_rhs = unsigned_pattern(UNSIGNED_LANES, 4)
    unsigned_lhs[0], unsigned_rhs[0] = np.uint32(0xFFFFFFFF), np.uint32(0xFFFFFFFF)
    unsigned_lhs[63], unsigned_rhs[63] = np.uint32(0x80000000), np.uint32(2)
    unsigned_lhs[64], unsigned_rhs[64] = np.uint32(0xFFFFFFFE), np.uint32(3)
    unsigned_lhs[127], unsigned_rhs[127] = np.uint32(0xDEADBEEF), np.uint32(0x10203040)

    lhs = np.concatenate((signed_lhs.view(np.uint32), unsigned_lhs))
    rhs = np.concatenate((signed_rhs.view(np.uint32), unsigned_rhs))
    golden_low = np.zeros(OUTPUT_ELEMS, dtype=np.uint32)
    golden_high = np.zeros(OUTPUT_ELEMS, dtype=np.uint32)

    signed_low, signed_high = product_halves(signed_lhs, signed_rhs, signed=True)
    golden_low[[0, 63]] = signed_low[[0, 63]]
    golden_high[[0, 63]] = signed_high[[0, 63]]
    unsigned_low, unsigned_high = product_halves(unsigned_lhs, unsigned_rhs, signed=False)
    golden_low[SIGNED_LANES:] = unsigned_low
    golden_high[SIGNED_LANES:] = unsigned_high

    assert (golden_low[0], golden_high[0]) == (0xFFFFFFFE, 0xFFFFFFFF)
    assert (golden_low[63], golden_high[63]) == (0x80000000, 0x00000000)
    assert (golden_low[64], golden_high[64]) == (0x00000001, 0xFFFFFFFE)
    assert (golden_low[127], golden_high[127]) == (0x00000000, 0x00000001)

    output_dir.mkdir(parents=True, exist_ok=True)
    lhs.tofile(output_dir / "lhs.bin")
    rhs.tofile(output_dir / "rhs.bin")
    np.zeros(OUTPUT_ELEMS, dtype=np.uint32).tofile(output_dir / "low.bin")
    np.zeros(OUTPUT_ELEMS, dtype=np.uint32).tofile(output_dir / "high.bin")
    golden_low.tofile(output_dir / "golden_low.bin")
    golden_high.tofile(output_dir / "golden_high.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
