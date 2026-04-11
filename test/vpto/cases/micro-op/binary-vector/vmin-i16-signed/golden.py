#!/usr/bin/env python3
# case: micro-op/binary-vector/vmin-i16-signed
# family: binary-vector
# target_ops: pto.vmin
# scenarios: core-i16-signed, full-mask
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(-1000, 1001, size=ELEMS, dtype=np.int16)
    v2 = rng.integers(-1000, 1001, size=ELEMS, dtype=np.int16)
    v3 = np.zeros(ELEMS, dtype=np.int16)
    golden_v3 = np.minimum(v1, v2)

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
