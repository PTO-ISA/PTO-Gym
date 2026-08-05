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

SOURCE_ELEMS = 512
LOGICAL_LANES = 300
BINS = 256


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    src = (np.arange(SOURCE_ELEMS, dtype=np.uint16) % BINS).astype(np.uint8)
    acc = (np.arange(BINS, dtype=np.uint16) % np.uint16(5)).astype(np.uint16)
    dst = np.full(BINS, np.uint16(0xcccc), dtype=np.uint16)

    counts = np.bincount(src[:LOGICAL_LANES].astype(np.int64), minlength=BINS)
    golden = (acc.astype(np.uint32) + counts.astype(np.uint32)).astype(np.uint16)

    src.tofile(output_dir / "v1.bin")
    acc.tofile(output_dir / "v2.bin")
    dst.tofile(output_dir / "v3.bin")
    golden.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
