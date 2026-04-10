#!/usr/bin/env python3
# case: micro-op/predicate-load-store/pldi-norm
# family: predicate-load-store
# target_ops: pto.pldi
# scenarios: packed-load, immediate-offset, representative-logical-elements

import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=np.uint8)
    output = np.fromfile("v2.bin", dtype=np.uint8)
    if golden.size < 256 or output.size < 256:
        print(
            f"[ERROR] Packed buffer too small: golden={golden.size} out={output.size}"
        )
        raise SystemExit(2)
    if not np.array_equal(golden[:256], output[:256]):
        diff = np.nonzero(golden[:256] != output[:256])[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (pldi NORM -> vsel): idx={idx} "
            f"golden={int(golden[idx])} out={int(output[idx])}"
        )
        raise SystemExit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
