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

ELEMS = 1024


def i32(value: int) -> np.int32:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return np.int32(value)


def split_i64(value: int) -> tuple[np.int32, np.int32]:
    value &= 0xFFFFFFFFFFFFFFFF
    return i32(value), i32(value >> 32)


def prmt(a: int, b: int, selector: int) -> np.int32:
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    src = [(b >> (8 * i)) & 0xFF for i in range(4)]
    src += [(a >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for pos in range(4):
        nibble = (selector >> (4 * pos)) & 0xF
        out |= src[nibble & 0x7] << (8 * pos)
    return i32(out)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(ELEMS, -1, dtype=np.int32)
    golden_v1 = np.zeros(ELEMS, dtype=np.int32)

    a = -2_000_000_000
    b = 3
    ua = 0xF0000000
    ub = 16
    x64 = 0xFFFFFFFFFFFFFFFF
    y64 = 2
    signed_prod = a * b
    unsigned_prod = ua * ub
    hi64_u = (x64 * y64) >> 64
    sx64 = -1
    sy64 = 2
    hi64_s = (sx64 * sy64) >> 64

    wide_s_lo, wide_s_hi = split_i64(signed_prod)
    wide_u_lo, wide_u_hi = split_i64(unsigned_prod)
    hi64_u_lo, hi64_u_hi = split_i64(hi64_u)
    hi64_s_lo, hi64_s_hi = split_i64(hi64_s)
    prmt_a = 0x11223344
    prmt_b = 0xAABBCCDD

    golden_v1[:14] = np.array(
        [
            i32(signed_prod >> 32),
            i32(unsigned_prod >> 32),
            wide_s_lo,
            wide_s_hi,
            wide_u_lo,
            wide_u_hi,
            hi64_u_lo,
            hi64_u_hi,
            prmt(prmt_a, prmt_b, 0x3210),
            prmt(prmt_a, prmt_b, 0x7654),
            prmt(prmt_a, prmt_b, 0x89AB),
            prmt(prmt_a, prmt_b, 0xFFFF),
            hi64_s_lo,
            hi64_s_hi,
        ],
        dtype=np.int32,
    )
    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
