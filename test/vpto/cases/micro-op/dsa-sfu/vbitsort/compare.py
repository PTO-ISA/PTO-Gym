#!/usr/bin/env python3
# case: micro-op/dsa-sfu/vbitsort
# family: dsa-sfu
# target_ops: pto.vbitsort
# scenarios: index-generation, layout-transform

import os
import sys

import numpy as np


def compare_bin(golden_path: str, output_path: str) -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=np.uint32)
    output = np.fromfile(output_path, dtype=np.uint32)
    return golden.shape == output.shape and np.array_equal(golden, output)


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v3.bin", "v3.bin")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
