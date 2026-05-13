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


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lhs = (np.arange(40 * 50, dtype=np.float16).reshape(40, 50) * np.float16(0.5) +
           np.float16(17)).astype(np.float16)
    rhs = (np.arange(50 * 64, dtype=np.float16).reshape(50, 64) * np.float16(0.25) +
           np.float16(3)).astype(np.float16)
    golden = lhs.astype(np.float32) @ rhs.astype(np.float32)

    lhs.reshape(-1).tofile(output_dir / "v1.bin")
    rhs.reshape(-1).tofile(output_dir / "v2.bin")
    outputs = {
        3: golden[:20, :],
        4: golden[20:, :],
        5: golden[:, :32],
        6: golden[:, 32:],
    }
    for index, value in outputs.items():
        np.zeros_like(value, dtype=np.float32).reshape(-1).tofile(output_dir / f"v{index}.bin")
        value.astype(np.float32).reshape(-1).tofile(output_dir / f"golden_v{index}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
