#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Merged vdiv golden data generator.
import argparse
from pathlib import Path
import numpy as np

ROWS = 32
COLS = 32
SEED = 19
LOGICAL_ELEMS = 1000
OUT_SENTINEL = np.float32(-123.25)

def f32_to_bf16_bits(values):
    wide = values.astype(np.float32, copy=False).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((wide >> 16) & np.uint32(1))
    return ((wide + rounding) >> 16).astype(np.uint16)

def bf16_bits_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)

def gen_f32(out, rng):
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
    v2 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
    g = (np.divide(v1, v2, where=v2!=0, out=np.full_like(v1, np.nan))).astype(np.float32, copy=False)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)
    v1.reshape(-1).tofile(out / "v1.bin")
    v2.reshape(-1).tofile(out / "v2.bin")
    v3.reshape(-1).tofile(out / "v3.bin")
    g.reshape(-1).tofile(out / "golden_v3.bin")

def gen_f16(out, rng):
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float16)
    v2 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float16)
    g = ((v1.astype(np.float32) / np.maximum(np.abs(v2.astype(np.float32)), 1e-8) * np.sign(v2.astype(np.float32))).astype(np.float16)).astype(np.float16)
    v3 = np.zeros((ROWS, COLS), dtype=np.float16)
    v1.reshape(-1).tofile(out / "v1_f16.bin")
    v2.reshape(-1).tofile(out / "v2_f16.bin")
    v3.reshape(-1).tofile(out / "v3_f16.bin")
    g.reshape(-1).tofile(out / "golden_v3_f16.bin")

def gen_f32_exceptional(out, rng):
    specials_a = np.array([-np.inf, -7.5, -0.0, 0.0, 1.0, np.inf, np.nan, 3.5], dtype=np.float32)
    specials_b = np.array([np.inf, 2.5, 0.0, -0.0, -1.0, -np.inf, 1.0, np.nan], dtype=np.float32)
    v1 = np.resize(specials_a, ROWS * COLS).reshape(ROWS, COLS).astype(np.float32)
    v2 = np.resize(specials_b, ROWS * COLS).reshape(ROWS, COLS).astype(np.float32)
    g = (np.divide(v1, v2, where=v2!=0, out=np.full_like(v1, np.nan))).astype(np.float32, copy=False)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)
    v1.reshape(-1).tofile(out / "v1_f32_exceptional.bin")
    v2.reshape(-1).tofile(out / "v2_f32_exceptional.bin")
    v3.reshape(-1).tofile(out / "v3_f32_exceptional.bin")
    g.reshape(-1).tofile(out / "golden_v3_f32_exceptional.bin")

def gen_tail(out, rng):
    v1 = rng.random((ROWS, COLS), dtype=np.float32)
    v2 = rng.random((ROWS, COLS), dtype=np.float32)
    v3 = np.full((ROWS, COLS), OUT_SENTINEL, dtype=np.float32)
    g = np.full((ROWS, COLS), OUT_SENTINEL, dtype=np.float32)
    g.reshape(-1)[:LOGICAL_ELEMS] = (np.divide(v1.reshape(-1)[:LOGICAL_ELEMS], v2.reshape(-1)[:LOGICAL_ELEMS], where=v2.reshape(-1)[:LOGICAL_ELEMS]!=0, out=np.full(LOGICAL_ELEMS, np.nan))).astype(np.float32, copy=False)
    v1.reshape(-1).tofile(out / "v1_tail.bin")
    v2.reshape(-1).tofile(out / "v2_tail.bin")
    v3.reshape(-1).tofile(out / "v3_tail.bin")
    g.reshape(-1).tofile(out / "golden_v3_tail.bin")

GENERATORS = [
    gen_f32,
    gen_f16,
    gen_f32_exceptional,
    gen_tail,
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for gen in GENERATORS:
        gen(out, rng)

if __name__ == "__main__":
    main()
