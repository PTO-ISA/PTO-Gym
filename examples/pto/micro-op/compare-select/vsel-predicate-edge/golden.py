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


LANES = 64
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lhs = rng.uniform(-8.0, 8.0, size=(LANES,)).astype(np.float32)
    rhs = lhs.copy()

    lane_ids = np.arange(LANES, dtype=np.int32)
    edge_mask = ((lane_ids < 4) | (lane_ids >= 60) | ((lane_ids % 17) == 0))
    rhs[edge_mask] = (rhs[edge_mask] + np.float32(3.5)).astype(np.float32)
    rhs[~edge_mask] = (rhs[~edge_mask] - np.float32(2.0)).astype(np.float32)

    out = np.zeros((LANES,), dtype=np.float32)
    golden = np.where(lhs > rhs, lhs, rhs).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    lhs.tofile(output_dir / "v1.bin")
    rhs.tofile(output_dir / "v2.bin")
    out.tofile(output_dir / "v3.bin")
    golden.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inputs/golden for VPTO vsel-predicate-edge.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
