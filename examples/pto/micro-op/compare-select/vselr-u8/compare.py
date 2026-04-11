#!/usr/bin/env python3
# case: micro-op/compare-select/vselr-u8
# family: compare-select
# target_ops: pto.vselr
# scenarios: core-u8, full-mask, explicit-lane-index

import os
import sys

import numpy as np


def compare_tensor(golden_path: str, output_path: str) -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    if golden.shape != output.shape:
        return False
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] Mismatch: idx={idx} golden={int(golden[idx])} out={int(output[idx])}")
        return False
    return True


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_tensor("golden_v3.bin", "v3.bin")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
