#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vsts-pk-b16
# family: vector-load-store
# target_ops: pto.vsts
# scenarios: core-i16, full-mask, aligned, dist-pk-b16
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


OUTPUT_BUFFER_BYTES = 4096
TOTAL_ELEMS_I16 = OUTPUT_BUFFER_BYTES // 2
# This case kernel only iterates 0..1024 on i16 lanes, so only 1024 packed bytes
# are semantically writable by vsts(pk_b16) in this testcase.
ACTIVE_ELEMS = 1024
LANES = 128
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-(2**15), 2**15, size=(TOTAL_ELEMS_I16,), dtype=np.int16)
    v2 = rng.integers(0, 256, size=(OUTPUT_BUFFER_BYTES,), dtype=np.uint8)
    golden_v2 = v2.copy()

    # PK_B16: write low 8 bits of each active b16 element as a compact byte stream.
    # Destination address is unchanged for non-post-update form; within each 256B
    # lane chunk only the first 128B are overwritten.
    v1_u16 = v1.view(np.uint16)
    packed_bytes_per_chunk = LANES
    for offset in range(0, ACTIVE_ELEMS, LANES):
        src = v1_u16[offset : offset + LANES]
        packed = (src & 0x00FF).astype(np.uint8)
        dst_byte_base = offset * 2
        golden_v2[dst_byte_base : dst_byte_base + packed_bytes_per_chunk] = packed

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vsts PK_B16 validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
