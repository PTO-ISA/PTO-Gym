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

ROWS = 4
COLS = 128
SCALE_SLOTS = 128
FP8_MAX = np.float32(448.0)
SCALE_LIMIT = np.float32(0.25)
SCALES = np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float32)
SENTINEL_F32 = np.float32(-777.0)
SENTINEL_U8 = np.uint8(0xA5)

Q_VALUES = np.array(
    [
        0.0,
        224.0,
        -224.0,
        112.0,
        -112.0,
        64.0,
        -64.0,
        32.0,
        -32.0,
        16.0,
        -16.0,
        8.0,
        -8.0,
        4.0,
        -4.0,
        2.0,
        -2.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
    ],
    dtype=np.float32,
)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_f32(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.uint32) << 16).view(np.float32)


def decode_f8e4m3fn(byte: int) -> np.float32:
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    mant = byte & 0x07
    if byte in (0x7F, 0xFF):
        return np.float32(np.nan)
    if exp == 0:
        return np.float32(sign * (mant / 8.0) * (2.0**-6))
    return np.float32(sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))


def f8e4m3fn_exact_bytes(values: np.ndarray) -> np.ndarray:
    exact = {}
    for byte in range(0x100):
        decoded = decode_f8e4m3fn(byte)
        if not np.isnan(decoded):
            exact.setdefault(np.float32(decoded).item(), byte)
    return np.array([exact[np.float32(value).item()] for value in values], dtype=np.uint8)


def f8e4m3fn_saturating_bytes(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -FP8_MAX, FP8_MAX)
    return f8e4m3fn_exact_bytes(clipped)


def generate(output_dir: Path) -> None:
    repeats = (COLS + len(Q_VALUES) - 1) // len(Q_VALUES)
    q_row = np.tile(Q_VALUES, repeats)[:COLS].astype(np.float32)

    src = np.empty((ROWS, COLS), dtype=np.uint16)
    golden_scale = np.full(SCALE_SLOTS, SENTINEL_F32, dtype=np.float32)
    golden_out = np.empty((ROWS, COLS), dtype=np.uint8)
    for row in range(ROWS):
        src[row] = f32_to_bf16_bits(q_row * SCALES[row])
        x_f32 = bf16_bits_to_f32(src[row])
        raw_scale = np.max(np.abs(x_f32)).astype(np.float32) / FP8_MAX
        scale = np.minimum(raw_scale, SCALE_LIMIT).astype(np.float32)
        golden_scale[row] = scale
        golden_out[row] = f8e4m3fn_saturating_bytes(x_f32 / scale)

    scale = np.full(SCALE_SLOTS, SENTINEL_F32, dtype=np.float32)
    out = np.full((ROWS, COLS), SENTINEL_U8, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    scale.tofile(output_dir / "v2.bin")
    out.tofile(output_dir / "v3.bin")
    golden_scale.tofile(output_dir / "golden_v2.bin")
    golden_out.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
