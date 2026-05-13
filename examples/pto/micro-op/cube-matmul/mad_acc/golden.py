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
    a = (((row * 3 + col * 5) % 17) - 8).astype(np.float16) / np.float16(4.0)
    k_idx = np.arange(K, dtype=np.float32).reshape(K, 1)
    n_idx = np.arange(N, dtype=np.float32).reshape(1, N)
    b = (((k_idx * 7 - n_idx * 2) % 19) - 9).astype(np.float16) / np.float16(5.0)
    a_acc = (((row * 2 - col * 7) % 13) - 6).astype(np.float16) / np.float16(3.0)
    b_acc = (((k_idx * 11 + n_idx * 3) % 17) - 8).astype(np.float16) / np.float16(4.0)
    c = np.zeros((M, N), dtype=np.float32)
    golden_c = a.astype(np.float32) @ b.astype(np.float32)
    golden_c += a_acc.astype(np.float32) @ b_acc.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    c.reshape(-1).tofile(output_dir / "v3.bin")
    a_acc.reshape(-1).tofile(output_dir / "v4.bin")
    b_acc.reshape(-1).tofile(output_dir / "v5.bin")
    golden_c.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
