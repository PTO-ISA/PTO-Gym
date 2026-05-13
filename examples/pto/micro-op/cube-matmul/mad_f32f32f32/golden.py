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

M = 16
N = 16
K = 16


def generate(output_dir: Path) -> None:
    row = np.arange(M, dtype=np.float32).reshape(M, 1)
    col = np.arange(K, dtype=np.float32).reshape(1, K)
    a = (((row * 11 + col * 3) % 31) - 15).astype(np.float32) / 16.0
    k_idx = np.arange(K, dtype=np.float32).reshape(K, 1)
    n_idx = np.arange(N, dtype=np.float32).reshape(1, N)
    b = (((k_idx * 5 - n_idx * 13) % 37) - 18).astype(np.float32) / 17.0

    a[0, 0] = np.float32(np.inf)
    b[0, 0] = np.float32(1.0)
    a[1, 1] = np.float32(np.nan)
    b[1, 1] = np.float32(1.0)
    a[2, :] = np.float32(0.0)
    b[:, 2] = np.float32(0.0)
    a[2, 2] = np.float32(2.0e38)
    b[2, 2] = np.float32(2.0)

    c_sat = np.zeros((M, N), dtype=np.float32)
    c_nosat = np.zeros((M, N), dtype=np.float32)
    saturated_a = np.nan_to_num(
        a,
        nan=np.float32(0.0),
        posinf=np.finfo(np.float32).max,
        neginf=-np.finfo(np.float32).max,
    ).astype(np.float32)
    saturated_b = np.nan_to_num(
        b,
        nan=np.float32(0.0),
        posinf=np.finfo(np.float32).max,
        neginf=-np.finfo(np.float32).max,
    ).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        f32_max = np.finfo(np.float32).max
        golden_sat = (saturated_a.astype(np.float64) @ saturated_b.astype(np.float64))
        golden_sat = np.nan_to_num(
            np.clip(golden_sat, -f32_max, f32_max),
            nan=0.0,
            posinf=f32_max,
            neginf=-f32_max,
        ).astype(np.float32)
        golden_nosat = (a @ b).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    c_sat.reshape(-1).tofile(output_dir / "v3.bin")
    c_nosat.reshape(-1).tofile(output_dir / "v4.bin")
    golden_sat.reshape(-1).tofile(output_dir / "golden_v3.bin")
    golden_nosat.reshape(-1).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
