#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
LANES = 64
def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)

    v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v2 = np.zeros((ROWS, COLS), dtype=np.float32)
    flat_in = v1.reshape(-1)
    flat_out = golden_v2.reshape(-1)
    group_elems = 8
    for offset in range(0, flat_in.size, LANES):
        chunk = flat_in[offset:offset + LANES]
        for gi, group in enumerate(range(0, chunk.size, group_elems)):
            # VCGMAX writes one reduced value per 32B block continuously to low lanes.
            flat_out[offset + gi] = np.max(chunk[group:group + group_elems])


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
