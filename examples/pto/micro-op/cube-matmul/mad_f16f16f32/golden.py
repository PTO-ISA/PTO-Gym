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

M = 1
N = 16
K = 32


def generate(output_dir: Path) -> None:
    row = np.arange(M, dtype=np.float32).reshape(M, 1)
    col = np.arange(K, dtype=np.float32).reshape(1, K)
    a = (((row * 3 + col * 5) % 17) - 8).astype(np.float16) / np.float16(4.0)
    k_idx = np.arange(K, dtype=np.float32).reshape(K, 1)
    n_idx = np.arange(N, dtype=np.float32).reshape(1, N)
    b = (((k_idx * 7 - n_idx * 2) % 19) - 9).astype(np.float16) / np.float16(5.0)
    c_default = np.zeros((M, N), dtype=np.float32)
    c_disable = np.zeros((M, N), dtype=np.float32)
    golden = a.astype(np.float32) @ b.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    c_default.reshape(-1).tofile(output_dir / "v3.bin")
    c_disable.reshape(-1).tofile(output_dir / "v4.bin")
    golden.reshape(-1).tofile(output_dir / "golden_v3.bin")
    golden.reshape(-1).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
