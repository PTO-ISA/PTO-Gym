#!/usr/bin/env python3
# case: micro-op/dsa-sfu/vmull
# family: dsa-sfu
# target_ops: pto.vmull
# scenarios: widening-op, hi-lo-split
# NOTE: bulk-generated coverage skeleton.

import os
import sys
import numpy as np


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    return golden.shape == output.shape and np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True)


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v2.bin", "v2.bin", np.int32, 0)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
