#!/usr/bin/env python3

import os
import sys
import numpy as np


def compare_bin(golden_path, output_path, dtype):
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        return False
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] Mismatch: idx={idx} golden={golden[idx]} out={output[idx]}")
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v3.bin", "v3.bin", np.int16)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
