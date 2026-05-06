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


LANES = 128
MASK_BYTES = 32
SEED = 19


def load_us_source_bits(packed: np.ndarray) -> np.ndarray:
    # `plds ..., "US"` on this path consumes only the first VL/16 bytes.
    bits = np.unpackbits(packed[:16], bitorder="little")
    return bits.astype(np.bool_, copy=False)


def plds_us_to_mask_b8(src_bits: np.ndarray) -> np.ndarray:
    # "US": duplicate each loaded bit once.
    return np.repeat(src_bits, 2).astype(np.bool_, copy=False)


def pbitcast_b8_to_b16(mask_b8: np.ndarray) -> np.ndarray:
    # Reinterpret the same predicate image at b16 granularity.
    # For the duplicated "US" image, each b16 lane observes the first bit of
    # the corresponding 2-bit pair, which reconstructs the original 128 bits.
    return mask_b8[::2].astype(np.bool_, copy=False)


def pintlv_b16_with_all(mask_b16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Interleave `%mask_b16` with `all_b16`, then split into low/high outputs.
    interleaved = np.empty((256,), dtype=np.bool_)
    interleaved[0::2] = mask_b16
    interleaved[1::2] = True
    return interleaved[:128], interleaved[128:]


def pbitcast_b16_to_b32(mask_b16_image: np.ndarray) -> np.ndarray:
    # Reinterpret the same predicate image at b32 granularity.
    # The b32 lanes read back the even-positioned b16 lanes.
    return mask_b16_image[0::2][:64].astype(np.bool_, copy=False)


def build_vsel_lanes_from_mask_pipeline(packed: np.ndarray) -> np.ndarray:
    src_bits = load_us_source_bits(packed)
    mask_b8 = plds_us_to_mask_b8(src_bits)
    mask_b16 = pbitcast_b8_to_b16(mask_b8)
    mask0_b16, mask1_b16 = pintlv_b16_with_all(mask_b16)
    mask0_b32 = pbitcast_b16_to_b32(mask0_b16)
    mask1_b32 = pbitcast_b16_to_b32(mask1_b16)
    return np.concatenate([mask0_b32, mask1_b32]).astype(np.bool_, copy=False)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-3.0, 3.0, size=(LANES,)).astype(np.float32)
    v2 = rng.uniform(-3.0, 3.0, size=(LANES,)).astype(np.float32)
    packed = rng.integers(0, 256, size=(MASK_BYTES,), dtype=np.uint8)
    lanes = build_vsel_lanes_from_mask_pipeline(packed)
    v4 = np.zeros((LANES,), dtype=np.float32)
    golden_v4 = np.where(lanes, v1, v2).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    packed.tofile(output_dir / "v3.bin")
    v4.tofile(output_dir / "v4.bin")
    golden_v4.tofile(output_dir / "golden_v4.bin")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Generate inputs/golden for VPTO vsel-f32-plds-us-pintlv-pbitcast."
  )
  parser.add_argument("--output-dir", type=Path, default=Path("."))
  parser.add_argument("--seed", type=int, default=SEED)
  args = parser.parse_args()
  generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
