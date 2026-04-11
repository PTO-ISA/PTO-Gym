#!/usr/bin/env python3
# case: micro-op/vec-scalar/vadds-i16-unsigned
# family: vec-scalar
# target_ops: pto.vadds
# scenarios: core-i16-unsigned, full-mask, scalar-operand

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SEED = 19
SCALAR = np.uint16(37)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    v1 = rng.integers(0, 60000, size=ELEMS, dtype=np.uint16)
    v2 = np.zeros(ELEMS, dtype=np.uint16)
    golden_v2 = (v1.astype(np.uint32) + int(SCALAR)).astype(np.uint16)

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
