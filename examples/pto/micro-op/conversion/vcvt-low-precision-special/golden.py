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

BYTES = 1024


def tiled(values):
    return np.resize(np.array(values, dtype=np.uint8), BYTES)


def bits_to_f32(bits):
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def f32_to_bits(value):
    return np.array([value], dtype=np.float32).view(np.uint32)[0]


def f32_to_bf16_bits(value):
    bits = int(f32_to_bits(value))
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return np.uint16((rounded >> 16) & 0xFFFF)


def decode_f8e4m3fn(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    mant = byte & 0x07
    if byte in (0x7F, 0xFF):
        return np.float32(np.nan)
    if exp == 0:
        return np.float32(sign * (mant / 8.0) * (2.0 ** -6))
    return np.float32(sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))


def decode_f8e5m2(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 2) & 0x1F
    mant = byte & 0x03
    if exp == 0x1F:
        if mant == 0:
            return np.float32(sign * np.inf)
        return np.float32(np.nan)
    if exp == 0:
        return np.float32(sign * (mant / 4.0) * (2.0 ** -14))
    return np.float32(sign * (1.0 + mant / 4.0) * (2.0 ** (exp - 15)))


def fp32_constructor(sign, exp, mant):
    return ((sign & 1) << 31) | ((exp & 0xFF) << 23) | (mant & 0x7FFFFF)


def decode_hif8(byte):
    if byte == 0x00:
        return bits_to_f32(0x00000000)
    if byte == 0x80:
        return np.float32(np.nan)
    if byte == 0x6F:
        return bits_to_f32(0x7F800000)
    if byte == 0xEF:
        return bits_to_f32(0xFF800000)

    input_sign = (byte >> 7) & 0x01
    bit6 = (byte >> 6) & 0x01
    bit5 = (byte >> 5) & 0x01
    bit4 = (byte >> 4) & 0x01
    bit3 = (byte >> 3) & 0x01
    if bit6 == 0 and bit5 == 0 and bit4 == 0 and bit3 == 0:
        return bits_to_f32(fp32_constructor(input_sign, (byte & 0x7) - 23 + 127, 0))

    if bit6 == 0:
        if bit5 == 0:
            if bit4 == 0:
                exp_width = 0
                d_width = 4
            else:
                exp_width = 1
                d_width = 3
        else:
            exp_width = 2
            d_width = 2
    else:
        exp_width = 3 if bit5 == 0 else 4
        d_width = 2
    man_width = 8 - d_width - exp_width - 1

    exp_mask = (1 << exp_width) - 1
    exp = 0
    if exp_width != 0:
        exp = ((byte >> man_width) & exp_mask) | (1 << (exp_width - 1))
        exp_msb = (byte >> (man_width + exp_width - 1)) & 0x1
        if exp_msb != 0:
            exp = -exp
    exp += 127

    man_mask = (1 << man_width) - 1
    mant = (byte & man_mask) << (23 - man_width)
    return bits_to_f32(fp32_constructor(input_sign, exp, mant))


FP4E1M2_TO_BF16 = np.array(
    [
        0x0000, 0x3E80, 0x3F00, 0x3F40, 0x3F80, 0x3FA0, 0x3FC0, 0x3FE0,
        0x8000, 0xBE80, 0xBF00, 0xBF40, 0xBF80, 0xBFA0, 0xBFC0, 0xBFE0,
    ],
    dtype=np.uint16,
)

FP4E2M1_TO_BF16 = np.array(
    [
        0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
        0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
    ],
    dtype=np.uint16,
)


def expected_f8(data, decoder):
    expected = np.zeros(256, dtype=np.float32)
    for part in range(4):
        start = part * 256
        selected = data[start + part : start + 256 : 4]
        values = [decoder(int(byte)) for byte in selected]
        expected[part * 64 : (part + 1) * 64] = np.array(values, dtype=np.float32)
    return expected


def expected_f4(data, table):
    expected = np.zeros(512, dtype=np.uint16)
    for part in range(4):
        start = part * 256
        selected = data[start + part : start + 256 : 4]
        values = []
        for byte in selected:
            values.append(table[int(byte) & 0x0F])
            values.append(table[(int(byte) >> 4) & 0x0F])
        expected[part * 128 : (part + 1) * 128] = np.array(values, dtype=np.uint16)
    return expected


def write_f32_quantize_case(output_dir, input_name, golden_name, pairs):
    values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), 64)
    expected_bytes = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), 64)
    input_values = np.tile(values, 4)
    expected = np.zeros(BYTES, dtype=np.uint8)
    for part in range(4):
        start = part * 256
        expected[start + part : start + 256 : 4] = expected_bytes
    input_values.tofile(output_dir / input_name)
    expected.tofile(output_dir / golden_name)


def write_bf16_quantize_case(output_dir, input_name, golden_name, nibble_pairs):
    values = np.resize(np.array([bits for bits, _ in nibble_pairs], dtype=np.uint16), 128)
    nibbles = np.resize(np.array([nibble for _, nibble in nibble_pairs], dtype=np.uint8), 128)
    packed = np.array([nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, 128, 2)], dtype=np.uint8)
    input_values = np.tile(values, 4)
    expected = np.zeros(BYTES, dtype=np.uint8)
    for part in range(4):
        start = part * 256
        expected[start + part : start + 256 : 4] = packed
    input_values.tofile(output_dir / input_name)
    expected.tofile(output_dir / golden_name)


def f8e4_quantize_pairs():
    exact = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x3C, 0xBC, 0x7E, 0xFE]
    pairs = [(decode_f8e4m3fn(byte), byte) for byte in exact]
    # This kernel uses #sat=1. In V300 instruction-controlled saturation,
    # infinities/overflow clamp to max finite and NaN is saturated to zero.
    pairs.extend(
        [
            (np.float32(np.inf), 0x7E),
            (np.float32(-np.inf), 0xFE),
            (np.float32(np.nan), 0x00),
            (np.float32(1000.0), 0x7E),
            (np.float32(-1000.0), 0xFE),
            (np.float32(1.0625), 0x38),
            (np.float32(1.1875), 0x3A),
            (np.float32(-1.0625), 0xB8),
            (np.float32(-1.1875), 0xBA),
        ]
    )
    return pairs


def f8e5_quantize_pairs():
    exact = [0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84, 0x3C, 0xBC, 0x40, 0xC0, 0x7B, 0xFB]
    pairs = [(decode_f8e5m2(byte), byte) for byte in exact]
    # This kernel uses #sat=1. In V300 instruction-controlled saturation,
    # infinities/overflow clamp to max finite and NaN is saturated to zero.
    pairs.extend(
        [
            (np.float32(np.inf), 0x7B),
            (np.float32(-np.inf), 0xFB),
            (np.float32(np.nan), 0x00),
            (np.float32(1.0e10), 0x7B),
            (np.float32(-1.0e10), 0xFB),
            (np.float32(1.125), 0x3C),
            (np.float32(1.375), 0x3E),
            (np.float32(-1.125), 0xBC),
            (np.float32(-1.375), 0xBE),
        ]
    )
    return pairs


def hif8_quantize_pairs():
    exact = [
        0x00, 0x80, 0x6F, 0xEF, 0x01, 0x81, 0x07, 0x87,
        0x08, 0x88, 0x10, 0x90, 0x18, 0x98, 0x20, 0xA0,
        0x40, 0xC0, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
    ]
    return [(decode_hif8(byte), byte) for byte in exact]


def f4e1_quantize_pairs():
    pairs = [(bits, nibble) for nibble, bits in enumerate(FP4E1M2_TO_BF16)]
    pairs.extend(
        [
            (f32_to_bf16_bits(0.625), 0x2),
            (f32_to_bf16_bits(0.75), 0x3),
            (f32_to_bf16_bits(0.875), 0x4),
            (f32_to_bf16_bits(-0.625), 0xA),
            (f32_to_bf16_bits(-0.75), 0xB),
            (f32_to_bf16_bits(-0.875), 0xC),
        ]
    )
    return pairs


def f4e2_quantize_pairs():
    pairs = [(bits, nibble) for nibble, bits in enumerate(FP4E2M1_TO_BF16)]
    pairs.extend(
        [
            (f32_to_bf16_bits(0.75), 0x2),
            (f32_to_bf16_bits(1.25), 0x2),
            (f32_to_bf16_bits(1.75), 0x4),
            (f32_to_bf16_bits(-0.75), 0xA),
            (f32_to_bf16_bits(-1.25), 0xA),
            (f32_to_bf16_bits(-1.75), 0xC),
        ]
    )
    return pairs


def generate(output_dir):
    inputs = {
        "f8e4": tiled(
            [
                0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88,
                0x38, 0xB8, 0x3C, 0xBC, 0x78, 0xF8, 0x7E, 0xFE,
                0x7F, 0xFF, 0x10, 0x90, 0x20, 0xA0, 0x70, 0xF0,
            ]
        ),
        "f8e5": tiled(
            [
                0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84,
                0x3C, 0xBC, 0x40, 0xC0, 0x7B, 0xFB, 0x7C, 0xFC,
                0x7D, 0xFD, 0x7F, 0xFF, 0x10, 0x90, 0x70, 0xF0,
            ]
        ),
        "hif8": tiled(
            [
                0x00, 0x80, 0x6F, 0xEF, 0x01, 0x81, 0x07, 0x87,
                0x08, 0x88, 0x10, 0x90, 0x18, 0x98, 0x20, 0xA0,
                0x40, 0xC0, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
            ]
        ),
        "f4e1": tiled(
            [
                0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
                0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
            ]
        ),
        "f4e2": tiled(
            [
                0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
                0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
            ]
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs["f8e4"].tofile(output_dir / "v1_f8e4_in.bin")
    inputs["f8e5"].tofile(output_dir / "v2_f8e5_in.bin")
    inputs["hif8"].tofile(output_dir / "v3_hif8_in.bin")
    inputs["f4e1"].tofile(output_dir / "v4_f4e1_in.bin")
    inputs["f4e2"].tofile(output_dir / "v5_f4e2_in.bin")

    np.full(BYTES, 0xA5, dtype=np.uint8).tofile(output_dir / "v6_f8e4_f32_out.bin")
    np.full(BYTES, 0xA5, dtype=np.uint8).tofile(output_dir / "v7_f8e5_f32_out.bin")
    np.full(BYTES, 0xA5, dtype=np.uint8).tofile(output_dir / "v8_hif8_f32_out.bin")
    np.full(BYTES, 0xA5, dtype=np.uint8).tofile(output_dir / "v9_f4e1_bf16_out.bin")
    np.full(BYTES, 0xA5, dtype=np.uint8).tofile(output_dir / "v10_f4e2_bf16_out.bin")

    expected_f8(inputs["f8e4"], decode_f8e4m3fn).tofile(output_dir / "golden_v6_f8e4_f32_out.bin")
    expected_f8(inputs["f8e5"], decode_f8e5m2).tofile(output_dir / "golden_v7_f8e5_f32_out.bin")
    expected_f8(inputs["hif8"], decode_hif8).tofile(output_dir / "golden_v8_hif8_f32_out.bin")
    expected_f4(inputs["f4e1"], FP4E1M2_TO_BF16).tofile(output_dir / "golden_v9_f4e1_bf16_out.bin")
    expected_f4(inputs["f4e2"], FP4E2M1_TO_BF16).tofile(output_dir / "golden_v10_f4e2_bf16_out.bin")

    write_f32_quantize_case(
        output_dir, "v11_f8e4_f32_in.bin", "golden_v16_f8e4_out.bin", f8e4_quantize_pairs()
    )
    write_f32_quantize_case(
        output_dir, "v12_f8e5_f32_in.bin", "golden_v17_f8e5_out.bin", f8e5_quantize_pairs()
    )
    write_f32_quantize_case(
        output_dir, "v13_hif8_f32_in.bin", "golden_v18_hif8_out.bin", hif8_quantize_pairs()
    )
    write_bf16_quantize_case(
        output_dir, "v14_f4e1_bf16_in.bin", "golden_v19_f4e1_out.bin", f4e1_quantize_pairs()
    )
    write_bf16_quantize_case(
        output_dir, "v15_f4e2_bf16_in.bin", "golden_v20_f4e2_out.bin", f4e2_quantize_pairs()
    )
    for name in [
        "v16_f8e4_out.bin",
        "v17_f8e5_out.bin",
        "v18_hif8_out.bin",
        "v19_f4e1_out.bin",
        "v20_f4e2_out.bin",
    ]:
        np.full(BYTES, 0xA5, dtype=np.uint8).tofile(output_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
