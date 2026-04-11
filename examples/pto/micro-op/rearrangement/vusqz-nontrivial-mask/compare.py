#!/usr/bin/env python3
# case: micro-op/rearrangement/vusqz-nontrivial-mask
# family: rearrangement
# target_ops: pto.vusqz
# scenarios: predicate-driven-rearrangement, prefix-count

import sys
import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v3.bin", dtype=np.int32)
    output = np.fromfile("v3.bin", dtype=np.int32)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        sys.exit(2)
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch at idx={idx}: golden={int(golden[idx])} out={int(output[idx])}"
        )
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
