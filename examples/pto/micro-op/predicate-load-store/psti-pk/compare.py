#!/usr/bin/env python3
# case: micro-op/predicate-load-store/psti-pk
# family: predicate-load-store
# target_ops: pto.psti
# scenarios: packed-store, immediate-offset, representative-logical-elements

import numpy as np


EXPECTED_WORDS = 8


def main() -> None:
    golden = np.fromfile("golden_v1.bin", dtype=np.uint32)
    output = np.fromfile("v1.bin", dtype=np.uint32)
    if golden.size != EXPECTED_WORDS or output.size != EXPECTED_WORDS:
      print(
          f"[ERROR] Unexpected word count: golden={golden.size} "
          f"out={output.size} expected={EXPECTED_WORDS}"
      )
      raise SystemExit(2)
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (psti PK raw packed store): idx={idx} "
            f"golden={int(golden[idx])} out={int(output[idx])}"
        )
        raise SystemExit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
