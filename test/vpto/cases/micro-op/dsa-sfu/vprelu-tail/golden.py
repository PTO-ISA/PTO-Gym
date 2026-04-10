#!/usr/bin/env python3
# case: micro-op/dsa-sfu/vprelu-tail
# family: dsa-sfu
# target_ops: pto.vprelu
# scenarios: core-f32, vector-alpha, tail-mask
# NOTE: bulk-generated coverage skeleton.
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
    v2 = rng.uniform(0.05, 0.5, size=(ROWS, COLS)).astype(np.float32)
    v3 = np.zeros((ROWS, COLS), dtype=np.float32)
    golden_v3 = np.where(v1 >= 0.0, v1, v1 * v2).astype(np.float32, copy=False)
    golden_v3.reshape(-1)[1000:] = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    v3.reshape(-1).tofile(output_dir / "v3.bin")
    golden_v3.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
