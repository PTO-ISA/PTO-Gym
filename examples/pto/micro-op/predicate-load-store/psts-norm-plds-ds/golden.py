#!/usr/bin/env python3
# case: micro-op/predicate-load-store/psts-norm-plds-ds
# family: predicate-load-store
# target_ops: pto.plds, pto.psts
# scenarios: predicate-load-store-composition, dynamic-offset, load-store-pair-preservation, representative-logical-elements

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from _predicate_load_store_case import norm_ds_compose, prefix_bits, write_case


SEED = 19
ACTIVE_BITS = 175


def generate(output_dir: Path, seed: int) -> None:
    del seed
    write_case(output_dir, norm_ds_compose(prefix_bits(ACTIVE_BITS)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inputs/golden for psts-norm-plds-ds.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
