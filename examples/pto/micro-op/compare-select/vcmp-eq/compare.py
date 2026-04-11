#!/usr/bin/python3

import os
import sys
import numpy as np


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    golden = np.fromfile("golden_v3.bin", dtype=np.uint8)
    output = np.fromfile("v3.bin", dtype=np.uint8)
    ok = golden.size >= 32 and output.size >= 32 and np.array_equal(golden[:32], output[:32])
    if not ok:
        if golden.size and output.size:
            diff = np.nonzero(golden[:32] != output[:32])[0]
            idx = int(diff[0]) if diff.size else 0
            print(f"[ERROR] Mismatch: idx={idx} golden={int(golden[idx])} out={int(output[idx])}")
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
