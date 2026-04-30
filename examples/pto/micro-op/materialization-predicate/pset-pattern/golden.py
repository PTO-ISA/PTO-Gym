#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/materialization-predicate/pset-pattern
# family: materialization-predicate
# target_ops: pto.pset_b16, pto.pset_b32, pto.pset_b8
# scenarios: pattern-mask, pat-all, pat-vl
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


SEED = 19
OUTPUT_WORDS = 24


def _pack_pset_prefix(active_lanes: int, bit_stride: int, store_bytes: int) -> np.ndarray:
    out = np.zeros((store_bytes,), dtype=np.uint8)
    for lane in range(active_lanes):
        bit_index = lane * bit_stride
        byte_index = bit_index // 8
        bit_in_byte = bit_index % 8
        out[byte_index] |= np.uint8(1 << bit_in_byte)
    return out


def generate(output_dir: Path, seed: int) -> None:
    del seed
    output_init = np.zeros((OUTPUT_WORDS,), dtype=np.uint32)

    out = np.zeros((OUTPUT_WORDS * 4,), dtype=np.uint8)
    out[0:32] = _pack_pset_prefix(active_lanes=256, bit_stride=1, store_bytes=32)
    out[32:48] = _pack_pset_prefix(active_lanes=8, bit_stride=2, store_bytes=16)
    out[64:80] = _pack_pset_prefix(active_lanes=16, bit_stride=4, store_bytes=16)
    golden = out.view(np.uint32)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_init.tofile(output_dir / "v1.bin")
    golden.tofile(output_dir / "golden_v1.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate packed predicate golden for VPTO micro-op pset-pattern validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
