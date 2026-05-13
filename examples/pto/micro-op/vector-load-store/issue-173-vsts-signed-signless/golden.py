#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/issue-173-vsts-signed-signless
# family: micro-op/vector-load-store
# target_ops: pto.vlds, pto.vsts
# scenarios: signed-i16, signless-i16, same-module, issue-173-regression

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 173


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    signed = rng.integers(-32768, 32768, size=ELEMS, dtype=np.int16)
    signless = rng.integers(-32768, 32768, size=ELEMS, dtype=np.int16)

    signed[:16] = np.array(
        [-32768, -30000, -12345, -1, 0, 1, 2, 3, 7, 15, 127, 255, 1024, 12345, 30000, 32767],
        dtype=np.int16,
    )
    signless[:16] = np.array(
        [32767, 30000, 12345, 1024, 255, 127, 15, 7, 3, 2, 1, 0, -1, -12345, -30000, -32768],
        dtype=np.int16,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    signed.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.int16).tofile(output_dir / "v2.bin")
    signless.tofile(output_dir / "v3.bin")
    np.zeros(ELEMS, dtype=np.int16).tofile(output_dir / "v4.bin")
    signed.tofile(output_dir / "golden_v2.bin")
    signless.tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
