#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vlds-unpk-b16
# family: vector-load-store
# target_ops: pto.vlds
# scenarios: core-f16, full-mask, aligned, dist-unpk-b16

import argparse
from pathlib import Path

import numpy as np


INPUT_ELEMS = 1024
OUTPUT_ELEMS = 2048
SRC_CHUNK = 64
DST_CHUNK = 128
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    src = rng.uniform(-8.0, 8.0, size=INPUT_ELEMS).astype(np.float16)
    dst = np.zeros((OUTPUT_ELEMS,), dtype=np.float16)
    golden = np.zeros((OUTPUT_ELEMS,), dtype=np.float16)

    for src_base in range(0, INPUT_ELEMS, SRC_CHUNK):
        dst_base = src_base * 2
        golden[dst_base : dst_base + DST_CHUNK : 2] = src[src_base : src_base + SRC_CHUNK]

    output_dir.mkdir(parents=True, exist_ok=True)
    src.view(np.uint16).tofile(output_dir / "v1.bin")
    dst.view(np.uint16).tofile(output_dir / "v2.bin")
    golden.view(np.uint16).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vlds UNPK_B16 validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
