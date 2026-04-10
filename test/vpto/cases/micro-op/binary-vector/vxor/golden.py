#!/usr/bin/env python3
# case: micro-op/binary-vector/vxor
# family: binary-vector
# target_ops: pto.vxor
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
    v1 = rng.integers(0, 0x10000, size=ELEMS, dtype=np.uint16)
    v2 = rng.integers(0, 0x10000, size=ELEMS, dtype=np.uint16)
    v3 = np.zeros(ELEMS, dtype=np.uint16)
    golden_v3 = np.bitwise_xor(v1, v2).astype(np.uint16, copy=False)

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
