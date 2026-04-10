#!/usr/bin/env python3
# case: micro-op/dsa-sfu/vci
# family: dsa-sfu / conversion
# target_ops: pto.vci
# scenarios: index-generation
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    _ = seed
    v1 = np.zeros((ROWS, COLS), dtype=np.int32)
    v2 = np.zeros((ROWS, COLS), dtype=np.int32)
    golden_v2 = np.arange(ROWS * COLS, dtype=np.int32).reshape(ROWS, COLS)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    golden_v2.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
