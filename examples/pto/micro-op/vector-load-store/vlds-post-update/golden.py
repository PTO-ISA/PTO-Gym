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


ELEMENTS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.uniform(-8.0, 8.0, size=(ELEMENTS,)).astype(np.float32)
    output = np.zeros((ELEMENTS,), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(output_dir / "input.bin")
    output.tofile(output_dir / "output.bin")
    data.tofile(output_dir / "golden_output.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inputs/golden for VPTO vlds post-update validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
