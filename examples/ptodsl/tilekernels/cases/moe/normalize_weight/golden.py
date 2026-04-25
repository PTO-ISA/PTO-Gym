#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROWS = 17
NUM_TOPK = 8
SEED = 19


def generate(output_dir: Path, *, rows: int, num_topk: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.01, 1.0, size=(rows, num_topk)).astype(np.float32)
    denom = (np.float32(1e-20) + weights.sum(axis=1, dtype=np.float32)).reshape(rows, 1)
    normalized = weights / denom

    output_dir.mkdir(parents=True, exist_ok=True)
    weights.reshape(-1).tofile(output_dir / "topk_weights.bin")
    np.zeros((rows, 1), dtype=np.float32).reshape(-1).tofile(output_dir / "denominator.bin")
    np.zeros((rows, num_topk), dtype=np.float32).reshape(-1).tofile(output_dir / "normalized_weights.bin")
    denom.reshape(-1).tofile(output_dir / "golden_denominator.bin")
    normalized.astype(np.float32, copy=False).reshape(-1).tofile(output_dir / "golden_normalized_weights.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--num-topk", type=int, default=NUM_TOPK)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, rows=args.rows, num_topk=args.num_topk, seed=args.seed)


if __name__ == "__main__":
    main()
