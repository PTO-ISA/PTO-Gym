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

ROWS = 8
COLS = 128
RNG_SEED = 19


def generate(output_dir: Path) -> None:
    np.random.seed(RNG_SEED)
    src = np.random.uniform(low=-2, high=2, size=(ROWS, COLS)).astype(np.float32)
    row_min = np.min(src, axis=1, keepdims=True)
    row_max = np.max(src, axis=1, keepdims=True)
    scale = ((row_max - row_min) / np.float32(255.0)).astype(np.float32)
    inv_scale = np.where(scale != 0, np.float32(1.0) / scale, np.float32(0.0)).astype(np.float32)
    offset = np.clip(np.round(-row_min / scale), 0, 255).astype(np.float32)
    rounded = np.round(src * inv_scale + offset).astype(np.float32)
    golden = np.clip(rounded.astype(np.float16), 0, 255).astype(np.uint8)
    dst = np.full((ROWS, COLS), 0xA5, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    inv_scale.reshape(ROWS).tofile(output_dir / "v2.bin")
    offset.reshape(ROWS).tofile(output_dir / "v3.bin")
    dst.tofile(output_dir / "v4.bin")
    golden.tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
