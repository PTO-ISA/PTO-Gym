#!/usr/bin/env python3
# case: micro-op/binary-vector/vdiv-f16
# family: binary-vector
# target_ops: pto.vdiv
# scenarios: core-f16, full-mask
import os
import sys
import numpy as np


def compare_bin_prefix(golden_path, output_path, dtype, eps, count):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=dtype, count=count)
    output = np.fromfile(output_path, dtype=dtype, count=count)
    return golden.shape == output.shape and np.allclose(
        golden, output, atol=eps, rtol=eps, equal_nan=True
    )


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin_prefix("golden_v3.bin", "v3.bin", np.float16, 5e-3, 1000)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
