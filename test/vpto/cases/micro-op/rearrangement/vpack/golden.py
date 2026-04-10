#!/usr/bin/env python3
# case: micro-op/rearrangement/vpack
# family: rearrangement
# target_ops: pto.vpack
# scenarios: pack-unpack, narrowing, half-placement, zero-fill-other-half
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
ELEMS = ROWS * COLS
CHUNK = 64
OUTPUT_ELEMS = ELEMS * 4
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-(1 << 20), 1 << 20, size=ELEMS, dtype=np.int32)
    v2 = np.zeros(OUTPUT_ELEMS, dtype=np.uint16)
    golden_v2 = np.zeros(OUTPUT_ELEMS, dtype=np.uint16)

    narrowed = v1.astype(np.uint16, copy=False)
    lower_half = golden_v2[: ELEMS * 2]
    higher_half = golden_v2[ELEMS * 2 :]
    for chunk_base in range(0, ELEMS, CHUNK):
        chunk = narrowed[chunk_base : chunk_base + CHUNK]
        out_base = (chunk_base // CHUNK) * (CHUNK * 2)
        lower_half[out_base : out_base + CHUNK] = chunk
        higher_half[out_base + CHUNK : out_base + 2 * CHUNK] = chunk

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for VPTO micro-op vpack validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
