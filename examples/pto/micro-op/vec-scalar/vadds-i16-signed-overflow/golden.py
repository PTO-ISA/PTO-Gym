#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vec-scalar/vadds-i16-signed-overflow
# family: vec-scalar
# target_ops: pto.vadds
# scenarios: core-i16-signed, full-mask, scalar-operand, integer-overflow

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19
SCALAR = np.int16(1024)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-16000, 16000, size=ELEMS, dtype=np.int16)
    v1[:12] = np.array(
        [
            32767,
            32766,
            32760,
            32000,
            0,
            1,
            -1,
            -32768,
            -32767,
            -32000,
            12345,
            -12345,
        ],
        dtype=np.int16,
    )
    v2 = np.zeros(ELEMS, dtype=np.int16)
    golden_v2 = (v1.astype(np.int32) + int(SCALAR)).astype(np.int16)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
