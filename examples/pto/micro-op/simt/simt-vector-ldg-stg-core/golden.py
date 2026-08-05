#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate input data and golden output for merged vector ldg/stg test."""

import argparse
from pathlib import Path

import numpy as np

LANES = 2
OPS_PER_LANE = 2
ARITH_ELEMS = LANES * OPS_PER_LANE * 6  # 3 regions x 2 ops x 2 elements = 24
COPY_ELEMS = 4


def _fill_arithmetic_buffer(lane, values_a, values_b, buf, golden):
    """Fill A and B regions and compute golden C = A + B."""
    base = lane * OPS_PER_LANE
    a0_idx = base * 2
    a1_idx = (base + 1) * 2
    b0_idx = (base + 4) * 2
    b1_idx = (base + 5) * 2
    c0_idx = (base + 8) * 2
    c1_idx = (base + 9) * 2

    a0 = np.array(values_a[0], dtype=buf.dtype)
    a1 = np.array(values_a[1], dtype=buf.dtype)
    b0 = np.array(values_b[0], dtype=buf.dtype)
    b1 = np.array(values_b[1], dtype=buf.dtype)

    buf[a0_idx : a0_idx + 2] = a0
    buf[a1_idx : a1_idx + 2] = a1
    buf[b0_idx : b0_idx + 2] = b0
    buf[b1_idx : b1_idx + 2] = b1

    golden[a0_idx : a0_idx + 2] = a0
    golden[a1_idx : a1_idx + 2] = a1
    golden[b0_idx : b0_idx + 2] = b0
    golden[b1_idx : b1_idx + 2] = b1

    golden[c0_idx : c0_idx + 2] = a0 + b0
    golden[c1_idx : c1_idx + 2] = a1 + b1


def _f32_to_bf16(arr_f32):
    """Convert float32 array to bf16 (uint16, upper 16 bits of float32)."""
    u32 = arr_f32.view(np.uint32)
    # Round to nearest even (proper bf16 rounding)
    rounding = np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))
    return ((u32 + rounding) >> 16).astype(np.uint16)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Arithmetic: f32x2 ──
    f32x2 = np.full(ARITH_ELEMS, -1.0, dtype=np.float32)
    golden_f32x2 = np.full(ARITH_ELEMS, -1.0, dtype=np.float32)
    for lane in range(LANES):
        _fill_arithmetic_buffer(
            lane,
            values_a=[(1.0 + lane * 10, 2.0 + lane * 10),
                      (3.0 + lane * 10, 4.0 + lane * 10)],
            values_b=[(0.5 + lane, 1.5 + lane),
                      (2.5 + lane, 3.5 + lane)],
            buf=f32x2, golden=golden_f32x2,
        )
    f32x2.tofile(output_dir / "f32x2.bin")
    golden_f32x2.tofile(output_dir / "golden_f32x2.bin")

    # ── Arithmetic: f16x2 ──
    f16x2 = np.full(ARITH_ELEMS, -1.0, dtype=np.float16)
    golden_f16x2 = np.full(ARITH_ELEMS, -1.0, dtype=np.float16)
    for lane in range(LANES):
        _fill_arithmetic_buffer(
            lane,
            values_a=[(1.0 + lane * 10, 2.0 + lane * 10),
                      (3.0 + lane * 10, 4.0 + lane * 10)],
            values_b=[(0.5 + lane, 1.5 + lane),
                      (2.5 + lane, 3.5 + lane)],
            buf=f16x2, golden=golden_f16x2,
        )
    f16x2.tofile(output_dir / "f16x2.bin")
    golden_f16x2.tofile(output_dir / "golden_f16x2.bin")

    # ── Arithmetic: bf16x2 (compute in float32, convert to bf16 at the end) ──
    bf16_f32_in = np.full(ARITH_ELEMS, -1.0, dtype=np.float32)
    bf16_f32_golden = np.full(ARITH_ELEMS, -1.0, dtype=np.float32)
    for lane in range(LANES):
        _fill_arithmetic_buffer(
            lane,
            values_a=[(1.0 + lane * 10, 2.0 + lane * 10),
                      (3.0 + lane * 10, 4.0 + lane * 10)],
            values_b=[(0.5 + lane, 1.5 + lane),
                      (2.5 + lane, 3.5 + lane)],
            buf=bf16_f32_in, golden=bf16_f32_golden,
        )
    bf16x2 = _f32_to_bf16(bf16_f32_in)
    golden_bf16x2 = _f32_to_bf16(bf16_f32_golden)
    bf16x2.tofile(output_dir / "bf16x2.bin")
    golden_bf16x2.tofile(output_dir / "golden_bf16x2.bin")

    # ── Arithmetic: i16x2 ──
    i16x2 = np.full(ARITH_ELEMS, -1, dtype=np.int16)
    golden_i16x2 = np.full(ARITH_ELEMS, -1, dtype=np.int16)
    for lane in range(LANES):
        _fill_arithmetic_buffer(
            lane,
            values_a=[(10 + lane * 20, 20 + lane * 20),
                      (30 + lane * 20, 40 + lane * 20)],
            values_b=[(5 + lane, 15 + lane),
                      (25 + lane, 35 + lane)],
            buf=i16x2, golden=golden_i16x2,
        )
    i16x2.tofile(output_dir / "i16x2.bin")
    golden_i16x2.tofile(output_dir / "golden_i16x2.bin")

    # ── Arithmetic: i32x2 ──
    i32x2 = np.full(ARITH_ELEMS, -1, dtype=np.int32)
    golden_i32x2 = np.full(ARITH_ELEMS, -1, dtype=np.int32)
    for lane in range(LANES):
        _fill_arithmetic_buffer(
            lane,
            values_a=[(100 + lane * 200, 200 + lane * 200),
                      (300 + lane * 200, 400 + lane * 200)],
            values_b=[(50 + lane, 150 + lane),
                      (250 + lane, 350 + lane)],
            buf=i32x2, golden=golden_i32x2,
        )
    i32x2.tofile(output_dir / "i32x2.bin")
    golden_i32x2.tofile(output_dir / "golden_i32x2.bin")

    # ── Copy: hif8x2 (4 x uint16, 2 input + 2 output) ──
    hif8x2 = np.zeros(COPY_ELEMS, dtype=np.uint16)
    hif8x2[0] = 0xABCD
    hif8x2[1] = 0x1234
    hif8x2[2] = 0xCDCD
    hif8x2[3] = 0xCDCD
    golden_hif8x2 = np.copy(hif8x2)
    golden_hif8x2[2] = hif8x2[0]
    golden_hif8x2[3] = hif8x2[1]
    hif8x2.tofile(output_dir / "hif8x2.bin")
    golden_hif8x2.tofile(output_dir / "golden_hif8x2.bin")

    # ── Copy: i8x2 (4 vectors x 2 bytes = 8 bytes, stored as uint8) ──
    i8x2 = np.zeros(COPY_ELEMS * 2, dtype=np.uint8)
    i8x2[0] = 0x01; i8x2[1] = 0x02
    i8x2[2] = 0x11; i8x2[3] = 0x12
    i8x2[4] = 0xCD; i8x2[5] = 0xCD
    i8x2[6] = 0xCD; i8x2[7] = 0xCD
    golden_i8x2 = np.copy(i8x2)
    golden_i8x2[4] = i8x2[0]; golden_i8x2[5] = i8x2[1]
    golden_i8x2[6] = i8x2[2]; golden_i8x2[7] = i8x2[3]
    i8x2.tofile(output_dir / "i8x2.bin")
    golden_i8x2.tofile(output_dir / "golden_i8x2.bin")

    # ── Copy: fp8e4x2 (4 x uint16) ──
    fp8e4x2 = np.zeros(COPY_ELEMS, dtype=np.uint16)
    fp8e4x2[0] = 0x4251
    fp8e4x2[1] = 0x3A60
    fp8e4x2[2] = 0xCDCD
    fp8e4x2[3] = 0xCDCD
    golden_fp8e4x2 = np.copy(fp8e4x2)
    golden_fp8e4x2[2] = fp8e4x2[0]
    golden_fp8e4x2[3] = fp8e4x2[1]
    fp8e4x2.tofile(output_dir / "fp8e4x2.bin")
    golden_fp8e4x2.tofile(output_dir / "golden_fp8e4x2.bin")

    # ── Copy: fp8e5x2 (4 x uint16) ──
    fp8e5x2 = np.zeros(COPY_ELEMS, dtype=np.uint16)
    fp8e5x2[0] = 0x5210
    fp8e5x2[1] = 0x3B80
    fp8e5x2[2] = 0xCDCD
    fp8e5x2[3] = 0xCDCD
    golden_fp8e5x2 = np.copy(fp8e5x2)
    golden_fp8e5x2[2] = fp8e5x2[0]
    golden_fp8e5x2[3] = fp8e5x2[1]
    fp8e5x2.tofile(output_dir / "fp8e5x2.bin")
    golden_fp8e5x2.tofile(output_dir / "golden_fp8e5x2.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
