#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: kernels/flash-attention
# family: kernels
# target_ops: pto.mte_gm_l1_frac, pto.mte_l1_l0a, pto.mte_l1_l0b, pto.mad,
#   pto.mte_l0c_ub, pto.mte_gm_ub, pto.mte_ub_gm, pto.vlds, pto.vcmax,
#   pto.vdup, pto.vmax, pto.vexpdif, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv,
#   pto.vsts, pto.sync.set, pto.sync.wait
# scenarios: flash-attention, cube-qk, tiled-online-softmax, q32-k32-d8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
SEQ = 32
HEAD_DIM = 16
VALUE_DIM = 8
TILE = 16
SEED = 29


def online_flash_attention(q: np.ndarray, k_t: np.ndarray, value_t: np.ndarray) -> np.ndarray:
    out = np.zeros((ROWS, VALUE_DIM), dtype=np.float32)
    for row in range(ROWS):
        running_max = np.float32(-np.inf)
        running_sum = np.float32(0.0)
        running_acc = np.zeros((VALUE_DIM,), dtype=np.float32)
        for tile_start in range(0, SEQ, TILE):
            tile_end = tile_start + TILE
            tile_scores = (q[row, :].astype(np.float32) @
                           k_t[:, tile_start:tile_end].astype(np.float32)).astype(np.float32)
            tile_max = np.max(tile_scores).astype(np.float32)
            new_max = np.maximum(running_max, tile_max).astype(np.float32)
            old_scale = np.exp(running_max - new_max, dtype=np.float32)
            probs = np.exp(tile_scores - new_max, dtype=np.float32)
            tile_sum = np.sum(probs, dtype=np.float32)
            tile_acc = probs.astype(np.float32) @ value_t[:, tile_start:tile_end].T
            running_acc = old_scale * running_acc + tile_acc.astype(np.float32)
            running_sum = old_scale * running_sum + tile_sum
            running_max = new_max
        out[row, :] = running_acc / running_sum
    return out


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    q = rng.normal(loc=0.0, scale=0.35, size=(ROWS, HEAD_DIM)).astype(np.float32)
    k = rng.normal(loc=0.0, scale=0.35, size=(SEQ, HEAD_DIM)).astype(np.float32)

    row_pattern = np.linspace(-1.0, 1.0, ROWS, dtype=np.float32).reshape(ROWS, 1)
    seq_pattern = np.linspace(1.0, -1.0, SEQ, dtype=np.float32).reshape(SEQ, 1)
    feat_wave = np.sin(np.linspace(0.0, 2.3 * np.pi, HEAD_DIM, dtype=np.float32)).reshape(1, HEAD_DIM)
    feat_ramp = np.linspace(-0.75, 0.75, HEAD_DIM, dtype=np.float32).reshape(1, HEAD_DIM)
    tile_bias = np.concatenate(
        [
            np.full((TILE, 1), -0.35, dtype=np.float32),
            np.full((SEQ - TILE, 1), 0.35, dtype=np.float32),
        ],
        axis=0,
    )
    q += 0.28 * row_pattern * feat_wave + 0.12 * feat_ramp
    k += 0.30 * seq_pattern * feat_wave - 0.18 * tile_bias * feat_ramp
    k_t = k.T.copy()
    value_t = rng.normal(loc=0.0, scale=0.55, size=(VALUE_DIM, SEQ)).astype(np.float32)
    value_t += 0.10 * np.linspace(-1.0, 1.0, VALUE_DIM, dtype=np.float32).reshape(VALUE_DIM, 1)
    value_t += 0.08 * np.cos(np.linspace(0.0, 3.0 * np.pi, SEQ, dtype=np.float32)).reshape(1, SEQ)
    out = np.zeros((ROWS, VALUE_DIM), dtype=np.float32)
    golden = online_flash_attention(q, k_t, value_t)

    output_dir.mkdir(parents=True, exist_ok=True)
    q.reshape(-1).tofile(output_dir / "v1.bin")
    k_t.reshape(-1).tofile(output_dir / "v2.bin")
    value_t.reshape(-1).tofile(output_dir / "v3.bin")
    out.reshape(-1).tofile(output_dir / "v4.bin")
    np.array([SEQ], dtype=np.int32).tofile(output_dir / "v5.bin")
    np.array([ROWS], dtype=np.int32).tofile(output_dir / "v6.bin")
    golden.astype(np.float32, copy=False).reshape(-1).tofile(output_dir / "golden_v4.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
