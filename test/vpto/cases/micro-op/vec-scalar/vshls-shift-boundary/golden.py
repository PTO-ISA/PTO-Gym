#!/usr/bin/env python3
# case: micro-op/vec-scalar/vshls-shift-boundary
# family: vec-scalar
# target_ops: pto.vshls
# scenarios: core-i16-unsigned, full-mask, scalar-operand, shift-boundary

import argparse
from pathlib import Path

import numpy as np


ELEMS = 1024
SHIFT = 15
PATTERN = np.array(
    [0x0000, 0x0001, 0x0002, 0x0003, 0x7FFF, 0x8000, 0x8001, 0xFFFF],
    dtype=np.uint16,
)


def generate(output_dir: Path, seed: int) -> None:
    del seed
    repeats = ELEMS // PATTERN.size
    v1 = np.tile(PATTERN, repeats)
    v2 = np.zeros(ELEMS, dtype=np.uint16)
    golden_v2 = np.left_shift(v1.astype(np.uint32), SHIFT).astype(np.uint16)

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
