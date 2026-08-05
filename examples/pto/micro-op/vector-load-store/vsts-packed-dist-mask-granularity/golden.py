#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vsts-packed-dist-mask-granularity
# family: micro-op/vector-load-store
# target_ops: pto.vlds, pto.vsts
# scenarios: packed-store, pk-b64, pk4-b32, b32-mask-granularity

import argparse
from pathlib import Path

import numpy as np


I64_ELEMS = 1024
F32_ELEMS = 512
U8_ELEMS = 256
SEED = 910


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    edge = np.array(
        [
            -(1 << 31),
            -(1 << 24) - 3,
            -(1 << 24),
            -(1 << 24) + 1,
            -65537,
            -32768,
            -1,
            0,
            1,
            32767,
            65537,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 31) - 2,
            (1 << 31) - 1,
        ],
        dtype=np.int32,
    )
    i64_base = rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=I64_ELEMS, dtype=np.int32
    )
    i64_base[: edge.size] = edge
    v1 = i64_base.astype(np.int64)

    v3 = rng.integers(0, 256, size=U8_ELEMS, dtype=np.uint8)
    ramp = np.arange(U8_ELEMS, dtype=np.uint16)
    v3 = ((v3.astype(np.uint16) + ramp) & 0xFF).astype(np.uint8)

    golden_v2 = np.concatenate(
        [i64_base[offset : offset + 16] for offset in range(0, I64_ELEMS, 32)]
    ).astype(np.float32)
    golden_v4 = np.zeros(U8_ELEMS, dtype=np.uint8)
    golden_v4[: U8_ELEMS // 4] = v3[0::4]

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    np.zeros(F32_ELEMS, dtype=np.float32).tofile(output_dir / "v2.bin")
    v3.tofile(output_dir / "v3.bin")
    np.zeros(U8_ELEMS, dtype=np.uint8).tofile(output_dir / "v4.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")
    golden_v4.tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inputs/golden for packed vsts mask granularity validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
