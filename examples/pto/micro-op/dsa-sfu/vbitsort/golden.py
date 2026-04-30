#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/dsa-sfu/vbitsort
# family: dsa-sfu
# target_ops: pto.vbitsort
# scenarios: index-generation, layout-transform

import argparse
from pathlib import Path

import numpy as np


SEED = 19
PROPOSALS = 32


def generate(output_dir: Path, seed: int) -> None:
    _ = seed
    scores = np.array([
        3.5, -2.0, 7.0, 7.0, 1.5, 4.25, 0.0, 9.5,
        -8.0, 9.5, 2.0, 2.0, 6.0, 6.0, -1.0, 5.75,
        5.75, 4.25, 8.0, 8.0, 3.0, -4.5, 1.25, 1.25,
        10.0, 10.0, -3.0, 0.5, 12.0, 12.0, -7.0, 6.5,
    ], dtype=np.float32)
    indices = np.array([
        100, 203, 77, 88, 12, 45, 501, 9,
        333, 7, 900, 901, 31, 32, 400, 62,
        63, 46, 73, 74, 15, 16, 120, 121,
        5, 6, 700, 701, 1, 2, 808, 90,
    ], dtype=np.uint32)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_indices = indices[order]

    packed = np.empty(PROPOSALS * 2, dtype=np.uint32)
    packed[0::2] = sorted_scores.view(np.uint32)
    packed[1::2] = sorted_indices

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.tofile(output_dir / "v1.bin")
    indices.tofile(output_dir / "v2.bin")
    np.zeros(PROPOSALS * 2, dtype=np.uint32).tofile(output_dir / "v3.bin")
    packed.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
