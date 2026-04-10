#!/usr/bin/env python3
# case: micro-op/binary-vector/vdiv-f16
# family: binary-vector
# target_ops: pto.vdiv
# scenarios: core-f16, full-mask
import argparse
from pathlib import Path

import numpy as np


ROWS = 32
COLS = 32
SEED = 19
LOGICAL_ELEMS = 1000


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float16)
    v2_mag = rng.uniform(0.5, 4.0, size=(ROWS, COLS)).astype(np.float32)
    v2_sign = np.where(rng.integers(0, 2, size=(ROWS, COLS), dtype=np.int32) == 0,
                       np.float32(-1.0), np.float32(1.0))
    v2 = (v2_mag * v2_sign).astype(np.float16)
    v3 = np.zeros((ROWS, COLS), dtype=np.float16)
    golden_v3 = np.zeros((ROWS, COLS), dtype=np.float16)
    golden_v3.reshape(-1)[:LOGICAL_ELEMS] = (
        v1.reshape(-1)[:LOGICAL_ELEMS].astype(np.float32)
        / v2.reshape(-1)[:LOGICAL_ELEMS].astype(np.float32)
    ).astype(np.float16)

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
