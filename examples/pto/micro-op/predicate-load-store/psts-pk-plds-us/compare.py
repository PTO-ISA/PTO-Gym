#!/usr/bin/env python3
# case: micro-op/predicate-load-store/psts-pk-plds-us
# family: predicate-load-store
# target_ops: pto.plds, pto.psts
# scenarios: predicate-load-store-composition, dynamic-offset, load-store-pair-preservation, representative-logical-elements

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from _predicate_load_store_case import compare_norm_store


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_norm_store("golden_v3.bin", "v3.bin")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
