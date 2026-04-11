#!/usr/bin/env python3
# case: micro-op/predicate-load-store/psti-norm-pldi-ds
# family: predicate-load-store
# target_ops: pto.pldi, pto.psti
# scenarios: predicate-load-store-composition, immediate-offset, load-store-pair-preservation, representative-logical-elements

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from _predicate_load_store_case import norm_ds_compose, prefix_bits, write_case


SEED = 19
ACTIVE_BITS = 143


def generate(output_dir: Path, seed: int, src_elem_bytes: int) -> None:
    del seed
    del src_elem_bytes
    write_case(output_dir, norm_ds_compose(prefix_bits(ACTIVE_BITS)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numpy-based inputs/golden for psti-norm-pldi-ds."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where v1.bin/v2.bin/v3.bin/golden_v3.bin are written.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Numpy random seed.")
    parser.add_argument(
        "--src-elem-bytes",
        type=int,
        default=4,
        help="Unused compatibility option kept for the shared runner surface.",
    )
    args = parser.parse_args()
    generate(args.output_dir, args.seed, args.src_elem_bytes)


if __name__ == "__main__":
    main()
