#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# s32 -> s8 golden for the NOSAT case (Case 2: tail with partial last iter).
#
# Semantics per lane (NOSAT):
#   valid lane   [0 .. 999]:  dst = int8_t(src[i] & 0xFF)
#   invalid lane [1000 .. 1023]: dst = sentinel 0xA5 (pmode="merge" preserves it)
#
# The kernel iterates 4 x 256 lanes.  The last iteration writes only 232
# lanes (256 - 24) via a create_mask + merge store.  The compare check runs
# on the full 1024-byte buffer so both the payload and the sentinel tail
# get verified.
#
# Same NOSAT/SAT guardrail as Case 1: probe set is chosen such that the
# NOSAT golden differs from the SAT reference on the valid prefix.

import argparse
from pathlib import Path

import numpy as np

SRC_ELEMS = 1024
DST_ELEMS = 1024
VALID_ELEMS = 1000
SENTINEL = 0xA5


def _build_inputs() -> np.ndarray:
    probes_i32 = np.array(
        [
                0,     1,    -1,    42,   -42,   127,  -128,    64,
              128,   200,   255,   256,   257,   511,   512,  1000,
             -129,  -200,  -255,  -256,  -257,  -511,  -512, -1000,
             0x7FFFFFFF,
            -0x80000000,
             0x7FFFFF80,
            -0x7FFFFF80,
             0x0000FF80,
            -0x000000FF,
             0x12345678,
            -0x12345678,
        ],
        dtype=np.int64,
    ).astype(np.int32)
    assert probes_i32.size == 32
    # 1024 / 32 = 32 tiles; every 256-lane iteration sees the same distribution.
    src = np.tile(probes_i32, SRC_ELEMS // probes_i32.size).astype(np.int32)
    assert src.size == SRC_ELEMS
    return src


def _nosat_ref(src_i32: np.ndarray) -> np.ndarray:
    return src_i32.astype(np.int8)


def _sat_ref(src_i32: np.ndarray) -> np.ndarray:
    return np.clip(src_i32, -128, 127).astype(np.int8)


def generate(output_dir: Path) -> None:
    src_i32 = _build_inputs()

    nosat_full = _nosat_ref(src_i32)
    sat_full = _sat_ref(src_i32)

    valid_nosat = nosat_full[:VALID_ELEMS]
    valid_sat = sat_full[:VALID_ELEMS]
    if np.array_equal(valid_nosat, valid_sat):
        raise SystemExit(
            "[FATAL] NOSAT golden coincidentally equals SAT reference on the "
            "valid prefix; adjust probe set to include out-of-range inputs.")

    dst_init = np.full(DST_ELEMS, SENTINEL, dtype=np.uint8)

    golden_bytes = dst_init.copy()
    golden_bytes[:VALID_ELEMS] = valid_nosat.view(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src_i32.tofile(output_dir / "v1.bin")
    dst_init.tofile(output_dir / "v2.bin")
    golden_bytes.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
