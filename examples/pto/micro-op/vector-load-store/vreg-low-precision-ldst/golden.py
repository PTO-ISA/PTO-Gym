#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vreg-low-precision-ldst
# family: vector-load-store
# target_ops: pto.vlds, pto.vsts, pto.vldsx2, pto.vstsx2, pto.vsldb, pto.vsstb, pto.vldas, pto.vldus, pto.vstus, pto.vstas

import argparse
from pathlib import Path

import numpy as np


SEED = 37
VECTOR_BYTES = 256
TOTAL_BYTES = 11 * VECTOR_BYTES


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 256, size=TOTAL_BYTES, dtype=np.uint8)

    # Give every low-precision segment a distinct deterministic byte pattern.
    for segment in range(TOTAL_BYTES // VECTOR_BYTES):
        start = segment * VECTOR_BYTES
        stop = start + VECTOR_BYTES
        ramp = np.arange(VECTOR_BYTES, dtype=np.uint16)
        data[start:stop] = ((data[start:stop].astype(np.uint16) + ramp + 17 * segment) & 0xFF).astype(np.uint8)

    golden = np.zeros((TOTAL_BYTES,), dtype=np.uint8)
    golden[0:5 * VECTOR_BYTES] = data[0:5 * VECTOR_BYTES]
    golden[5 * VECTOR_BYTES:7 * VECTOR_BYTES] = data[5 * VECTOR_BYTES:7 * VECTOR_BYTES]
    # The b8 block load/store pair observes the next 32-byte block after a
    # same-stride roundtrip, matching the existing block-layout instruction contract.
    golden[7 * VECTOR_BYTES:8 * VECTOR_BYTES] = data[7 * VECTOR_BYTES + 32:8 * VECTOR_BYTES + 32]
    # Exercise the typed bf16 vsstb intrinsic with contiguous 32-byte blocks.
    golden[8 * VECTOR_BYTES:9 * VECTOR_BYTES] = data[8 * VECTOR_BYTES:9 * VECTOR_BYTES]
    # vldus starts from an explicitly unaligned base.
    golden[9 * VECTOR_BYTES:10 * VECTOR_BYTES] = data[9 * VECTOR_BYTES + 1:10 * VECTOR_BYTES + 1]
    # vstus/vstas makes only the explicit state-store offset bytes visible here.
    golden[10 * VECTOR_BYTES:10 * VECTOR_BYTES + 3] = data[10 * VECTOR_BYTES:10 * VECTOR_BYTES + 3]
    output_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(output_dir / "v1.bin")
    np.zeros((TOTAL_BYTES,), dtype=np.uint8).tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw byte inputs/golden for low-precision VPTO vreg load/store validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
