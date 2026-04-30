#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/unary-vector/vabs-i16-signed-overflow-edge
# family: unary-vector
# target_ops: pto.vabs
# scenarios: core-i16-signed, full-mask, integer-overflow

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.integers(-30000, 30000, size=ELEMS, dtype=np.int16)
    edge = np.array(
        [-32768, -32767, -12345, -1, 0, 1, 12345, 32767,
         -32768, -2, 2, -32766, 32766, -1024, 1024, -17],
        dtype=np.int16,
    )
    data[:edge.size] = edge
    golden = np.abs(data).astype(np.int16, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.int16).tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
