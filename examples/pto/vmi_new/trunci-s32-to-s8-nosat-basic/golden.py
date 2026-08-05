#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# s32 -> s8 golden for the NOSAT case (Case 1: basic, no tail).
#
# Semantics (NOSAT):
#   dst[i] = numpy.int32(src[i]).astype(numpy.int8)   # signed low-8 truncation
#          = int8_t(src[i] & 0xFF)                    # 2's complement wrap
#
# Semantics (SAT, reference only, MUST differ):
#   dst[i] = numpy.clip(src[i], -128, 127).astype(numpy.int8)
#
# Input is a curated 32-value probe set repeated to fill 256 lanes:
#   * ~1/4 lanes fit inside [-128, 127]   (SAT == NOSAT)
#   * ~1/4 lanes positive overflow        (SAT clips to 127, NOSAT wraps)
#   * ~1/4 lanes negative overflow        (SAT clips to -128, NOSAT wraps)
#   * ~1/4 lanes hit high-bit corners     (INT32_MAX / INT32_MIN / mixed)
#
# A guardrail assertion refuses to emit the artefacts if the resulting NOSAT
# golden happens to equal the SAT reference; that would let a broken lowering
# (which silently produced SAT) silently pass this case.

import argparse
from pathlib import Path

import numpy as np

SRC_ELEMS = 256
DST_ELEMS = 256


def _build_inputs() -> np.ndarray:
    probes_i32 = np.array(
        [
            # In-range: NOSAT == SAT
                0,     1,    -1,    42,   -42,   127,  -128,    64,
            # Positive overflow: NOSAT != SAT
              128,   200,   255,   256,   257,   511,   512,  1000,
            # Negative overflow: NOSAT != SAT
             -129,  -200,  -255,  -256,  -257,  -511,  -512, -1000,
            # High-bit corners (Python ints; cast to int32)
             0x7FFFFFFF,   # INT32_MAX  -> low byte 0xFF -> int8 -1
            -0x80000000,   # INT32_MIN  -> low byte 0x00 -> int8  0
             0x7FFFFF80,   #             -> low byte 0x80 -> int8 -128
            -0x7FFFFF80,   #             -> low byte 0x80 -> int8 -128
             0x0000FF80,   #             -> low byte 0x80 -> int8 -128
            -0x000000FF,   #             -> low byte 0x01 -> int8  1
             0x12345678,   #             -> low byte 0x78 -> int8 120
            -0x12345678,   #             -> low byte 0x88 -> int8 -120
        ],
        dtype=np.int64,   # int32 range would overflow -0x80000000 as a Python literal
    ).astype(np.int32)
    assert probes_i32.size == 32, f"probe count changed: {probes_i32.size}"
    src = np.tile(probes_i32, SRC_ELEMS // probes_i32.size).astype(np.int32)
    assert src.size == SRC_ELEMS
    return src


def _nosat_ref(src_i32: np.ndarray) -> np.ndarray:
    # numpy int32->int8 astype is defined as low-8 signed truncation.
    return src_i32.astype(np.int8)


def _sat_ref(src_i32: np.ndarray) -> np.ndarray:
    return np.clip(src_i32, -128, 127).astype(np.int8)


def generate(output_dir: Path) -> None:
    src_i32 = _build_inputs()
    golden = _nosat_ref(src_i32)
    sat_ref = _sat_ref(src_i32)

    # Guardrail: NOSAT and SAT MUST produce different bit patterns overall.
    # Otherwise a lowering bug that silently emits SAT could pass this case.
    if np.array_equal(golden, sat_ref):
        raise SystemExit(
            "[FATAL] NOSAT golden coincidentally equals SAT reference; "
            "adjust probe set to include out-of-range inputs.")

    dst_init = np.full(DST_ELEMS, 0xA5, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src_i32.tofile(output_dir / "v1.bin")
    dst_init.tofile(output_dir / "v2.bin")
    # Emit golden as raw bytes so compare.py can read as uint8 without endian
    # or sign surprises.
    golden.view(np.uint8).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
