#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
ALPHA = np.float32(0.125)


def generate(output_dir: Path, seed: int) -> None:
    del seed
    specials = np.array(
        [-np.inf, -8.0, -1.0, -0.0, 0.0, 1.0, np.inf, np.nan],
        dtype=np.float32,
    )
    v1 = np.resize(specials, ROWS * COLS).reshape(ROWS, COLS).astype(np.float32)
    v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v2 = np.where(v1 >= 0.0, v1, v1 * ALPHA).astype(np.float32, copy=False)

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
