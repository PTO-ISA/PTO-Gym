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
K = 16
N = 16
DST_STRIDE = 48


def generate(output_dir: Path) -> None:
    a = np.zeros((M, K), dtype=np.float16)
    b = np.zeros((K, N), dtype=np.float16)
    c = np.zeros((M, DST_STRIDE), dtype=np.float32)
    golden_c = np.zeros((M, DST_STRIDE), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    c.reshape(-1).tofile(output_dir / "v3.bin")
    golden_c.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
