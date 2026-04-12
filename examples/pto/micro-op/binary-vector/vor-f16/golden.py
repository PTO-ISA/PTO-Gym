#!/usr/bin/env python3
# case: micro-op/binary-vector/vor-f16
# family: binary-vector
# target_ops: pto.vor
# scenarios: core-f16, full-mask
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bits1 = rng.integers(0, 0x10000, size=ELEMS, dtype=np.uint16)
    bits2 = rng.integers(0, 0x10000, size=ELEMS, dtype=np.uint16)
    bits1[:8] = np.array(
        [0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e00, 0x3555],
        dtype=np.uint16,
    )
    bits2[:8] = np.array(
        [0x0001, 0x0001, 0x4000, 0x2000, 0x0001, 0x0001, 0x0100, 0x0aaa],
        dtype=np.uint16,
    )
    v1 = bits1.view(np.float16)
    v2 = bits2.view(np.float16)
    v3 = np.zeros(ELEMS, dtype=np.float16)
    golden_v3 = np.bitwise_or(bits1, bits2).view(np.float16)

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
