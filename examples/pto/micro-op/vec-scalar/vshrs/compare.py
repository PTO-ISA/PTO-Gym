#!/usr/bin/env python3
# case: micro-op/vec-scalar/vshrs
# family: vec-scalar
# target_ops: pto.vshrs
# scenarios: core-i16-unsigned, full-mask, scalar-operand

import os
import sys
import numpy as np


def compare_bin(golden_path, output_path, dtype):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    return golden.shape == output.shape and np.array_equal(golden, output)


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v2.bin", "v2.bin", np.uint16)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
