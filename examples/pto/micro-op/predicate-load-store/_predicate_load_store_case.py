#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
OUTPUT_BYTES = ROWS * COLS
PREDICATE_BITS = 256
# For the current A5 predicate load/store surface used by these composition
# cases, the user-visible packed NORM footprint is 16 bytes. Bytes beyond that
# range are not part of the checked result footprint.
NORM_STORAGE_BYTES = 16


def prefix_bits(active_bits: int) -> np.ndarray:
    bits = np.zeros((PREDICATE_BITS,), dtype=np.uint8)
    bits[:active_bits] = 1
    return bits


def pk_us_compose(bits: np.ndarray) -> np.ndarray:
    packed = bits[::2]
    return np.repeat(packed, 2).astype(np.uint8, copy=False)


def norm_ds_compose(bits: np.ndarray) -> np.ndarray:
    source = np.concatenate(
        [bits.astype(np.uint8, copy=False), np.zeros_like(bits, dtype=np.uint8)]
    )
    return source[::2][:PREDICATE_BITS].astype(np.uint8, copy=False)


def norm_store_bytes(bits: np.ndarray) -> np.ndarray:
    packed = np.packbits(bits.astype(np.uint8, copy=False), bitorder="little")
    out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
    out[:NORM_STORAGE_BYTES] = packed[:NORM_STORAGE_BYTES]
    return out


def write_default_inputs(output_dir: Path) -> None:
    np.zeros((ROWS * COLS,), dtype=np.float32).tofile(output_dir / "v1.bin")
    np.zeros((ROWS * COLS,), dtype=np.float32).tofile(output_dir / "v2.bin")
    np.zeros((OUTPUT_BYTES,), dtype=np.uint8).tofile(output_dir / "v3.bin")


def write_case(output_dir: Path, bits: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_default_inputs(output_dir)
    norm_store_bytes(bits).tofile(output_dir / "golden_v3.bin")


def compare_norm_store(golden_path: str, output_path: str) -> bool:
    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    if golden.size < NORM_STORAGE_BYTES or output.size < NORM_STORAGE_BYTES:
        return False
    if not np.array_equal(golden[:NORM_STORAGE_BYTES], output[:NORM_STORAGE_BYTES]):
        diff = np.nonzero(golden[:NORM_STORAGE_BYTES] != output[:NORM_STORAGE_BYTES])[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (predicate load/store composition): idx={idx} "
            f"golden={int(golden[idx])} out={int(output[idx])}"
        )
        return False
    return True
