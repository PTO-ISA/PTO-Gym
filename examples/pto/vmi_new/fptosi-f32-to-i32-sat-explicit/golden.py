#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# f32 -> si32 (FpToSi) with saturate = "SAT" golden.
#
# V300 SAT policy for FpToSi (rounding = "R", i.e. RN):
#   * finite in-range values          -> round-to-nearest-even to int32
#   * finite out-of-range (positive)  -> INT32_MAX (0x7FFFFFFF)
#   * finite out-of-range (negative)  -> INT32_MIN (0x80000000)
#   * +inf                            -> INT32_MAX
#   * -inf                            -> INT32_MIN
#   * NaN                             -> 0
#
# We build a 256-lane vector that deliberately mixes:
#   - a "normal" band of small integers (RN round-half-to-even edges)
#   - out-of-range positives / negatives
#   - +inf / -inf / NaN
# so the case exercises every saturation branch, not just the trivial path.

import argparse
import struct
from pathlib import Path

import numpy as np

ELEMS = 256
INT32_MAX = np.int32(0x7FFFFFFF)
INT32_MIN = np.int32(-0x80000000)


def f32(x):
    return np.float32(x)


# A carefully chosen 32-value probe pattern.  ELEMS = 256 = 8 * 32, so tiling
# gives a clean 8x-repeat schedule that survives Packed4 lane grouping.
PROBE = [
    # --- in-range: RN round-half-to-even edges + typical values ---
    f32(0.0),          # 0
    f32(0.5),          # RN -> 0 (nearest-even)
    f32(1.5),          # RN -> 2
    f32(2.5),          # RN -> 2
    f32(-0.5),         # RN -> 0
    f32(-1.5),         # RN -> -2
    f32(-2.5),         # RN -> -2
    f32(1.0),          # 1
    f32(-1.0),         # -1
    f32(127.0),        # 127
    f32(-128.0),       # -128
    f32(32767.0),      # 32767
    f32(-32768.0),     # -32768

    # --- out-of-range positive: must clamp to INT32_MAX ---
    f32(2.147484e9),   # slightly above INT32_MAX (~2.147483648e9)
    f32(1.0e10),       # far above INT32_MAX
    f32(3.4e38),       # near f32 max
    f32(float("inf")), # +inf

    # --- out-of-range negative: must clamp to INT32_MIN ---
    f32(-2.147484e9),
    f32(-1.0e10),
    f32(-3.4e38),
    f32(float("-inf")),

    # --- NaN saturates to 0 under V300 SAT ---
    f32(float("nan")),

    # --- extra edge values to keep the probe balanced (32 entries) ---
    f32(-1.0),
    f32(0.25),          # RN -> 0
    f32(0.75),          # RN -> 1
    f32(1e6),
    f32(-1e6),
    f32(1234567.5),     # RN -> 1234568
    f32(-1234567.5),    # RN -> -1234568
    f32(2.147483583e9), # just below INT32_MAX (~2^31 - 65 as f32), representable
    f32(-2.147483648e9),# exactly -2^31 as f32 -> INT32_MIN
    f32(999999.0),
]

assert len(PROBE) == 32, f"PROBE length must be 32, got {len(PROBE)}"


def rn_to_int(x):
    # Python-side round-half-to-even, mirroring hardware RN on integer boundary.
    # np.rint uses banker's rounding, which matches RN.
    return int(np.rint(x))


def saturating_fptosi(v):
    if np.isnan(v):
        return 0
    if np.isposinf(v):
        return int(INT32_MAX)
    if np.isneginf(v):
        return int(INT32_MIN)
    # Round-to-nearest-even, then clamp.
    r = rn_to_int(v)
    if r >= int(INT32_MAX):
        return int(INT32_MAX)
    if r <= int(INT32_MIN):
        return int(INT32_MIN)
    return r


def generate(output_dir: Path) -> None:
    probe = np.array(PROBE, dtype=np.float32)
    src = np.tile(probe, ELEMS // len(probe))[:ELEMS].astype(np.float32)

    gold_seq = np.array([saturating_fptosi(v) for v in probe], dtype=np.int32)
    golden = np.tile(gold_seq, ELEMS // len(gold_seq))[:ELEMS].astype(np.int32)

    # Deliberately fill the pre-kernel destination with a sentinel so a
    # completely-missed store shows up as garbage instead of coincidence.
    dst = np.full(ELEMS, -559038737, dtype=np.int32)  # 0xDEADBEEF

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    dst.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()