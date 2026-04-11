#!/usr/bin/env python3
# case: micro-op/binary-vector/vadd-i16-unsigned-overflow
# family: binary-vector
# target_ops: pto.vadd
# scenarios: core-i16-unsigned, full-mask, integer-overflow

import os
import sys

import numpy as np


def compare_bin(golden_path: str, output_path: str, dtype) -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        return False
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    return golden.shape == output.shape and np.array_equal(golden, output)


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v3.bin", "v3.bin", np.uint16)
    if not ok:
      if strict:
          print("[ERROR] compare failed")
          sys.exit(2)
      print("[WARN] compare failed (non-gating)")
      return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
