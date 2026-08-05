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

WIDE_ELEMS = 256
DENSE_ELEMS = 128
SEED = 43
SENTINEL = np.float32(-1777.0)


def vintlv(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = lhs.size // 2
    low = np.empty_like(lhs)
    high = np.empty_like(lhs)
    low[0::2] = lhs[:half]
    low[1::2] = rhs[:half]
    high[0::2] = lhs[half:]
    high[1::2] = rhs[half:]
    return low, high


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lhs_f16 = rng.uniform(-100.0, 100.0, size=WIDE_ELEMS).astype(np.float16)
    rhs_f16 = rng.uniform(200.0, 400.0, size=WIDE_ELEMS).astype(np.float16)
    lhs = lhs_f16.astype(np.float32)
    rhs = rhs_f16.astype(np.float32)
    intlv_low, intlv_high = vintlv(lhs, rhs)
    wide_empty = np.full(WIDE_ELEMS, SENTINEL, dtype=np.float32)
    dense_lhs = rng.uniform(500.0, 700.0, size=DENSE_ELEMS).astype(np.float32)
    dense_rhs = rng.uniform(-700.0, -500.0, size=DENSE_ELEMS).astype(np.float32)
    dense_intlv_low, dense_intlv_high = vintlv(dense_lhs, dense_rhs)
    dense_empty = np.full(DENSE_ELEMS, SENTINEL, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    lhs_f16.tofile(output_dir / "v1.bin")
    rhs_f16.tofile(output_dir / "v2.bin")
    for index in range(3, 7):
        wide_empty.tofile(output_dir / f"v{index}.bin")
    dense_lhs.tofile(output_dir / "v7.bin")
    dense_rhs.tofile(output_dir / "v8.bin")
    dense_empty.tofile(output_dir / "v9.bin")
    dense_empty.tofile(output_dir / "v10.bin")
    intlv_low.tofile(output_dir / "golden_v3.bin")
    intlv_high.tofile(output_dir / "golden_v4.bin")
    lhs.tofile(output_dir / "golden_v5.bin")
    rhs.tofile(output_dir / "golden_v6.bin")
    dense_intlv_low.tofile(output_dir / "golden_v7.bin")
    dense_intlv_high.tofile(output_dir / "golden_v8.bin")
    dense_lhs.tofile(output_dir / "golden_v9.bin")
    dense_rhs.tofile(output_dir / "golden_v10.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
