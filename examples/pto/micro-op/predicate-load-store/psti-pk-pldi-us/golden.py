#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/predicate-load-store/psti-pk-pldi-us
# family: predicate-load-store
# target_ops: pto.pldi, pto.psti
# scenarios: predicate-load-store-composition, immediate-offset, load-store-pair-preservation, representative-logical-elements

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from _predicate_load_store_case import pk_us_compose, prefix_bits, write_case


SEED = 19
ACTIVE_BITS = 145


def generate(output_dir: Path, seed: int, src_elem_bytes: int) -> None:
    del seed
    del src_elem_bytes
    write_case(output_dir, pk_us_compose(prefix_bits(ACTIVE_BITS)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inputs/golden for psti-pk-pldi-us.")
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
