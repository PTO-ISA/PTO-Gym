#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/gather-scatter/vgather2
# family: gather-scatter
# target_ops: pto.vgather2
# scenarios: core-f32, core-f16-u16-offsets, core-u8-u16-offsets, core-i8-u16-offsets, full-mask, non-contiguous, explicit-index-pattern, load-effect-validation, no-alias
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    flat = rng.uniform(-8.0, 8.0, size=(ROWS * COLS,)).astype(np.float32)
    offsets = ((np.arange(ROWS * COLS, dtype=np.int32) * 17) + 3) % (ROWS * COLS)
    gathered = flat[offsets].reshape(ROWS, COLS)
    v1 = flat.reshape(ROWS, COLS)
    v2 = offsets.reshape(ROWS, COLS)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)

    flat_f16 = rng.uniform(-4.0, 4.0, size=(ROWS * COLS,)).astype(np.float16)
    offsets_u16 = (((np.arange(ROWS * COLS, dtype=np.uint32) * 11) + 5) %
                   (ROWS * COLS)).astype(np.uint16)
    gathered_f16 = flat_f16[offsets_u16.astype(np.int32)].reshape(ROWS, COLS)
    v4 = flat_f16.reshape(ROWS, COLS)
    v5 = offsets_u16.reshape(ROWS, COLS)
    v6 = np.zeros((ROWS, COLS), dtype=np.float16)

    flat_u8 = rng.integers(0, 256, size=(ROWS * COLS,), dtype=np.uint8)
    gathered_u8 = flat_u8[offsets_u16.astype(np.int32)].astype(np.uint16).reshape(ROWS, COLS)
    v7 = flat_u8.reshape(ROWS, COLS)
    v8 = np.zeros((ROWS, COLS), dtype=np.uint16)

    flat_i8 = rng.integers(-128, 128, size=(ROWS * COLS,), dtype=np.int16).astype(np.int8)
    gathered_i8 = flat_i8[offsets_u16.astype(np.int32)].view(np.uint8).astype(np.int16).reshape(ROWS, COLS)
    v9 = flat_i8.reshape(ROWS, COLS)
    v10 = np.zeros((ROWS, COLS), dtype=np.int16)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    v3.reshape(-1).tofile(output_dir / "v3.bin")
    v4.reshape(-1).tofile(output_dir / "v4.bin")
    v5.reshape(-1).tofile(output_dir / "v5.bin")
    v6.reshape(-1).tofile(output_dir / "v6.bin")
    v7.reshape(-1).tofile(output_dir / "v7.bin")
    v8.reshape(-1).tofile(output_dir / "v8.bin")
    v9.reshape(-1).tofile(output_dir / "v9.bin")
    v10.reshape(-1).tofile(output_dir / "v10.bin")
    gathered.reshape(-1).tofile(output_dir / "golden_v3.bin")
    gathered_f16.reshape(-1).tofile(output_dir / "golden_v6.bin")
    gathered_u8.reshape(-1).tofile(output_dir / "golden_v8.bin")
    gathered_i8.reshape(-1).tofile(output_dir / "golden_v10.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vgather2 validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
