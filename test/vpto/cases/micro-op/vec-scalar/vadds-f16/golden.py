#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19
SCALE = np.float16(1.5)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-4.0, 4.0, size=ELEMS).astype(np.float16)
    v2 = np.zeros(ELEMS, dtype=np.float16)
    golden_v2 = (v1.astype(np.float32) + np.float32(SCALE)).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
