#!/usr/bin/env python3
# case: micro-op/binary-vector/vshr-shift-boundary
# family: binary-vector
# target_ops: pto.vshr
# scenarios: core-i16-unsigned, full-mask
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(0, 1 << 16, size=ELEMS, dtype=np.uint16)
    shift_cycle = np.array([0, 1, 14, 15, 15, 14, 1, 0], dtype=np.uint16)
    v2 = np.resize(shift_cycle, ELEMS).astype(np.uint16, copy=False)
    v3 = np.zeros(ELEMS, dtype=np.uint16)
    golden_v3 = np.right_shift(v1, v2 & np.uint16(15)).astype(np.uint16, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    v3.tofile(output_dir / "v3.bin")
    golden_v3.tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
