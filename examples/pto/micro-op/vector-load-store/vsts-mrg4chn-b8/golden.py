#!/usr/bin/env python3
# case: micro-op/vector-load-store/vsts-mrg4chn-b8
# family: vector-load-store
# target_ops: pto.vsts
# scenarios: core-i8, full-mask, aligned, dist-mrg4chn-b8
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


TOTAL_ELEMS_I8 = 4096
ACTIVE_ELEMS = 1024
LANES = 256
BLOCK_ELEMS = 32  # 32B block on b8
SEED = 19


def _merge_4chn_b8_block(block):
    # Source is interleaved [c0_0, c1_0, c2_0, c3_0, c0_1, ...].
    # MRG4CHN_B8 stores channel-major [c0_0..c0_7, c1_0..c1_7, c2_0.., c3_0..].
    return np.concatenate((block[0::4], block[1::4], block[2::4], block[3::4]))


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-(2**7), 2**7, size=(TOTAL_ELEMS_I8,), dtype=np.int8)
    v2 = rng.integers(-(2**7), 2**7, size=(TOTAL_ELEMS_I8,), dtype=np.int8)
    golden_v2 = v2.copy()

    # Full mask case: each covered 32B block is fully overwritten after merge.
    for offset in range(0, ACTIVE_ELEMS, LANES):
        src_chunk = v1[offset : offset + LANES]
        dst_chunk = golden_v2[offset : offset + LANES]
        for blk in range(0, LANES, BLOCK_ELEMS):
            dst_chunk[blk : blk + BLOCK_ELEMS] = _merge_4chn_b8_block(
                src_chunk[blk : blk + BLOCK_ELEMS]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vsts MRG4CHN_B8 validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
