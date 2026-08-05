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

ELEMS = 1024
LANES = 32


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(ELEMS, -1, dtype=np.int32)
    golden_v1 = np.full(ELEMS, -1, dtype=np.int32)
    v1[0:LANES] = 10
    v1[32 : 32 + LANES] = 20
    v1[64 : 64 + LANES] = 10
    v1[96 : 96 + LANES] = 10
    golden_v1[0:LANES] = 15
    golden_v1[32 : 32 + LANES] = 20
    golden_v1[64 : 64 + LANES] = 42
    golden_v1[96 : 96 + LANES] = 2
    golden_v1[128 : 128 + LANES] = 10
    golden_v1[160 : 160 + LANES] = 20
    golden_v1[192 : 192 + LANES] = 10
    golden_v1[224 : 224 + LANES] = 10
    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
