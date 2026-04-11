#!/usr/bin/env python3
# case: micro-op/vector-load-store/vsts-mrg2chn-b16
# family: vector-load-store
# target_ops: pto.vsts
# scenarios: core-i16, full-mask, aligned, dist-mrg2chn-b16
# coding=utf-8

import os
import sys
import numpy as np

TOTAL_BYTES = 4096
ACTIVE_ELEMS = 1024
BYTES_PER_ELEM = 2


def build_checked_mask(total_bytes):
    # This case only covers the prefix [0, ACTIVE_ELEMS) on i16 elements.
    # For dist=MRG2CHN_B16 with full mask, compare only that writable prefix.
    writable_bytes = ACTIVE_ELEMS * BYTES_PER_ELEM
    mask = np.zeros((total_bytes,), dtype=bool)
    mask[:writable_bytes] = True
    return mask


def compare_bin(golden_path, output_path):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False

    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        return False

    if golden.size != TOTAL_BYTES:
        print(
            f"[ERROR] Unexpected byte size for this case: got {golden.size}, expected {TOTAL_BYTES}"
        )
        return False

    checked = build_checked_mask(golden.size)
    checked_golden = golden[checked]
    checked_output = output[checked]
    if not np.array_equal(checked_golden, checked_output):
        diff = np.nonzero(checked_golden != checked_output)[0]
        idx = int(diff[0]) if diff.size else 0
        global_idx = int(np.nonzero(checked)[0][idx]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (checked footprint): {golden_path} vs {output_path}, "
            f"first diff at checked_idx={idx}, global_idx={global_idx} "
            f"(golden=0x{int(checked_golden[idx]):02x}, out=0x{int(checked_output[idx]):02x})"
        )
        return False
    print(
        f"[INFO] compared writable footprint only: {int(np.count_nonzero(checked))}/{golden.size} bytes"
    )
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v2.bin", "v2.bin")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
