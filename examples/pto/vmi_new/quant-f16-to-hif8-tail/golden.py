#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Golden reference for Phase 1 path f16 -> hif8 with tail-mask store.
# The kernel processes 4 chunks of 256 lanes; the last-tail mask covers only
# the first LOGICAL_ELEMS lanes.  Untouched lanes must keep the SENTINEL byte.
#
# hif8 (HiFloat8) values here are chosen so the ExactValue set does not depend
# on the rnd=A vs rnd=H distinction: {0, +/-1, +/-2, +/-4} are exactly
# representable in hif8 without rounding, and sat=SAT never triggers on them.
# A5 HiFloat8 uses its own byte encoding, distinct from f8E4M3FN:
#   0.0   -> 0x00
#   +1.0  -> 0x08
#   -1.0  -> 0x88
#   +0.5  -> 0x18
#   +2.0  -> 0x10
#   -2.0  -> 0x90
#   +4.0  -> 0x20
#   -4.0  -> 0xA0

import argparse
from pathlib import Path

import numpy as np

ELEMS = 1024
LOGICAL_ELEMS = 1000
VALUES = np.array([0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 4.0, -4.0], dtype=np.float32)
HIF8_BYTES = np.array([0x00, 0x08, 0x88, 0x18, 0x10, 0x90, 0x20, 0xA0], dtype=np.uint8)
SENTINEL = np.uint8(0xA5)


def generate(output_dir: Path) -> None:
    repeats = (ELEMS + len(VALUES) - 1) // len(VALUES)
    src = np.tile(VALUES, repeats)[:ELEMS].astype(np.float16)
    packed = np.tile(HIF8_BYTES, repeats)[:ELEMS].astype(np.uint8)
    dst = np.full(ELEMS, SENTINEL, dtype=np.uint8)
    golden = np.full(ELEMS, SENTINEL, dtype=np.uint8)
    golden[:LOGICAL_ELEMS] = packed[:LOGICAL_ELEMS]

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
