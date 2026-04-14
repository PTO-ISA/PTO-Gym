#!/usr/bin/env python3
# case: micro-op/vector-load-store/vsts-mrg2chn-b16
# family: vector-load-store
# target_ops: pto.vsts
# scenarios: core-i16, full-mask, aligned, dist-mrg2chn-b16
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


TOTAL_BYTES = 4096
TOTAL_ELEMS_I16 = TOTAL_BYTES // 2
ACTIVE_ELEMS = 1024
LANES = 128
BLOCK_ELEMS = 16  # 32B block on b16
SEED = 19


def _merge_2chn_b16_block(block):
    # Source is interleaved [c0_0, c1_0, c0_1, c1_1, ...].
    # MRG2CHN_B16 stores channel-major [c0_0..c0_7, c1_0..c1_7].
    return np.concatenate((block[0::2], block[1::2]))


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-(2**15), 2**15, size=(TOTAL_ELEMS_I16,), dtype=np.int16)
    v2 = rng.integers(-(2**15), 2**15, size=(TOTAL_ELEMS_I16,), dtype=np.int16)
    golden_v2 = v2.copy()

    # Full mask case: every logical element is active in the covered prefix.
    for offset in range(0, ACTIVE_ELEMS, LANES):
        src_chunk = v1[offset : offset + LANES]
        dst_chunk = golden_v2[offset : offset + LANES]
        for blk in range(0, LANES, BLOCK_ELEMS):
            dst_chunk[blk : blk + BLOCK_ELEMS] = _merge_2chn_b16_block(
                src_chunk[blk : blk + BLOCK_ELEMS]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vsts MRG2CHN_B16 validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
