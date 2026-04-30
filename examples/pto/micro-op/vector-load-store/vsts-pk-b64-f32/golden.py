#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vsts-pk-b64-f32
# family: vector-load-store
# target_ops: pto.vsts
# scenarios: core-f32, full-mask, aligned, dist-pk-b64, width-agnostic-dist

import argparse
from pathlib import Path

import numpy as np


ELEMENTS = 1024
LANES = 64
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-8.0, 8.0, size=(ELEMENTS,)).astype(np.float32)
    v2 = rng.uniform(-8.0, 8.0, size=(ELEMENTS,)).astype(np.float32)
    golden_v2 = np.array(v2, copy=True)

    for offset in range(0, ELEMENTS, LANES):
        chunk = v1[offset : offset + LANES]
        packed = chunk[0::2]
        golden_v2[offset : offset + packed.size] = packed

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vsts PK_B64 on f32 validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
