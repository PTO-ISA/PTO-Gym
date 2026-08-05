#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Merged vadd golden data generator: 9 variants.
# Each variant writes uniquely suffixed .bin files.

import argparse
from pathlib import Path

import numpy as np

ROWS = 32
COLS = 32
SEED = 19
LOGICAL_ELEMS = 1000
OUT_SENTINEL = np.float32(-123.25)


# ---- helpers ----

def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    wide = values.astype(np.float32, copy=False).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((wide >> 16) & np.uint32(1))
    return ((wide + rounding) >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def wrap_add_i16(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    bits = lhs.view(np.uint16).astype(np.uint32) + rhs.view(np.uint16).astype(np.uint32)
    return (bits & 0xFFFF).astype(np.uint16).view(np.int16)


def wrap_add_u16(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    wide = lhs.astype(np.uint32) + rhs.astype(np.uint32)
    return (wide & 0xFFFF).astype(np.uint16)


# ---- generators ----

def gen_f32(out: Path, rng: np.random.Generator) -> None:
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
    v2 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
    g = (v1 + v2).astype(np.float32, copy=False)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)
    v1.reshape(-1).tofile(out / "v1.bin")
    v2.reshape(-1).tofile(out / "v2.bin")
    v3.reshape(-1).tofile(out / "v3.bin")
    g.reshape(-1).tofile(out / "golden_v3.bin")


def gen_f16(out: Path, rng: np.random.Generator) -> None:
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float16)
    v2 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float16)
    v3 = np.zeros((ROWS, COLS), dtype=np.float16)
    g = (v1.astype(np.float32) + v2.astype(np.float32)).astype(np.float16)
    v1.reshape(-1).tofile(out / "v1_f16.bin")
    v2.reshape(-1).tofile(out / "v2_f16.bin")
    v3.reshape(-1).tofile(out / "v3_f16.bin")
    g.reshape(-1).tofile(out / "golden_v3_f16.bin")


def gen_bf16(out: Path, rng: np.random.Generator) -> None:
    elems = ROWS * COLS
    v1_f32 = rng.uniform(-4.0, 4.0, size=elems).astype(np.float32)
    v2_f32 = rng.uniform(-4.0, 4.0, size=elems).astype(np.float32)
    v1 = f32_to_bf16_bits(v1_f32)
    v2 = f32_to_bf16_bits(v2_f32)
    v3 = np.zeros(elems, dtype=np.uint16)
    g = f32_to_bf16_bits(bf16_bits_to_f32(v1) + bf16_bits_to_f32(v2))
    v1.tofile(out / "v1_bf16.bin")
    v2.tofile(out / "v2_bf16.bin")
    v3.tofile(out / "v3_bf16.bin")
    g.tofile(out / "golden_v3_bf16.bin")


def gen_f32_exceptional(out: Path, _rng: np.random.Generator) -> None:
    specials_a = np.array([-np.inf, -7.5, -0.0, 0.0, 1.0, np.inf, np.nan, 3.5], dtype=np.float32)
    specials_b = np.array([np.inf, 2.5, 0.0, -0.0, -1.0, -np.inf, 1.0, np.nan], dtype=np.float32)
    v1 = np.resize(specials_a, ROWS * COLS).reshape(ROWS, COLS).astype(np.float32)
    v2 = np.resize(specials_b, ROWS * COLS).reshape(ROWS, COLS).astype(np.float32)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)
    g = (v1 + v2).astype(np.float32, copy=False)
    v1.reshape(-1).tofile(out / "v1_x.bin")
    v2.reshape(-1).tofile(out / "v2_x.bin")
    v3.reshape(-1).tofile(out / "v3_x.bin")
    g.reshape(-1).tofile(out / "golden_v3_x.bin")


def gen_i16_signed(out: Path, rng: np.random.Generator) -> None:
    v1 = rng.integers(-1000, 1001, size=(ROWS, COLS), dtype=np.int16)
    v2 = rng.integers(-1000, 1001, size=(ROWS, COLS), dtype=np.int16)
    v3 = np.zeros((ROWS, COLS), dtype=np.int16)
    g = (v1.astype(np.int32) + v2.astype(np.int32)).astype(np.int16)
    v1.reshape(-1).tofile(out / "v1_i16s.bin")
    v2.reshape(-1).tofile(out / "v2_i16s.bin")
    v3.reshape(-1).tofile(out / "v3_i16s.bin")
    g.reshape(-1).tofile(out / "golden_v3_i16s.bin")


def gen_i16_signed_overflow(out: Path, _rng: np.random.Generator) -> None:
    elems = ROWS * COLS
    lhs_pattern = np.array([32767, 32760, -32768, -32760, 1000, -1000, 12345, -12345], dtype=np.int16)
    rhs_pattern = np.array([1, 100, -1, -100, 30000, -30000, 23456, -23456], dtype=np.int16)
    repeats = elems // lhs_pattern.size
    v1 = np.tile(lhs_pattern, repeats)
    v2 = np.tile(rhs_pattern, repeats)
    v3 = np.zeros(elems, dtype=np.int16)
    g = wrap_add_i16(v1, v2)
    v1.tofile(out / "v1_i16s_ov.bin")
    v2.tofile(out / "v2_i16s_ov.bin")
    v3.tofile(out / "v3_i16s_ov.bin")
    g.tofile(out / "golden_v3_i16s_ov.bin")


def gen_i16_unsigned(out: Path, rng: np.random.Generator) -> None:
    v1 = rng.integers(0, 2001, size=(ROWS, COLS), dtype=np.uint16)
    v2 = rng.integers(0, 2001, size=(ROWS, COLS), dtype=np.uint16)
    v3 = np.zeros((ROWS, COLS), dtype=np.uint16)
    g = (v1.astype(np.uint32) + v2.astype(np.uint32)).astype(np.uint16)
    v1.reshape(-1).tofile(out / "v1_i16u.bin")
    v2.reshape(-1).tofile(out / "v2_i16u.bin")
    v3.reshape(-1).tofile(out / "v3_i16u.bin")
    g.reshape(-1).tofile(out / "golden_v3_i16u.bin")


def gen_i16_unsigned_overflow(out: Path, _rng: np.random.Generator) -> None:
    elems = ROWS * COLS
    lhs_pattern = np.array([65535, 65530, 65500, 60000, 100, 0, 32768, 12345], dtype=np.uint16)
    rhs_pattern = np.array([1, 10, 1000, 10000, 65535, 5, 40000, 60000], dtype=np.uint16)
    repeats = elems // lhs_pattern.size
    v1 = np.tile(lhs_pattern, repeats)
    v2 = np.tile(rhs_pattern, repeats)
    v3 = np.zeros(elems, dtype=np.uint16)
    g = wrap_add_u16(v1, v2)
    v1.tofile(out / "v1_i16u_ov.bin")
    v2.tofile(out / "v2_i16u_ov.bin")
    v3.tofile(out / "v3_i16u_ov.bin")
    g.tofile(out / "golden_v3_i16u_ov.bin")


def gen_tail(out: Path, rng: np.random.Generator) -> None:
    v1 = rng.random((ROWS, COLS), dtype=np.float32)
    v2 = rng.random((ROWS, COLS), dtype=np.float32)
    v3 = np.full((ROWS, COLS), OUT_SENTINEL, dtype=np.float32)
    g = np.full((ROWS, COLS), OUT_SENTINEL, dtype=np.float32)
    g.reshape(-1)[:LOGICAL_ELEMS] = (
        v1.reshape(-1)[:LOGICAL_ELEMS] + v2.reshape(-1)[:LOGICAL_ELEMS]
    ).astype(np.float32, copy=False)
    v1.reshape(-1).tofile(out / "v1_tail.bin")
    v2.reshape(-1).tofile(out / "v2_tail.bin")
    v3.reshape(-1).tofile(out / "v3_tail.bin")
    g.reshape(-1).tofile(out / "golden_v3_tail.bin")


# ---- main ----

GENERATORS = [
    gen_f32,
    gen_f16,
    gen_bf16,
    gen_f32_exceptional,
    gen_i16_signed,
    gen_i16_signed_overflow,
    gen_i16_unsigned,
    gen_i16_unsigned_overflow,
    gen_tail,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for gen in GENERATORS:
        gen(out, rng)


if __name__ == "__main__":
    main()
