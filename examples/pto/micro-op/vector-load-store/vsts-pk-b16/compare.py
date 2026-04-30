#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vsts-pk-b16
# family: vector-load-store
# target_ops: pto.vsts
# scenarios: core-i16, full-mask, aligned, dist-pk-b16
# coding=utf-8

import os
import sys
import numpy as np

OUTPUT_BUFFER_BYTES = 4096
# Keep this aligned with kernel.pto loop bound (offset: 0..1024 step 128 on i16).
ACTIVE_ELEMS = 1024
LANES = 128
BYTES_PER_ELEM = 2


def build_checked_mask(total_bytes):
    # For this case kernel:
    # - loop offset: 0..1024 step 128 (i16 elements)
    # - dist=PK_B16 stores 1 byte per active i16 element
    # So each iteration writes 128 bytes at dst_byte_base = offset * 2.
    mask = np.zeros((total_bytes,), dtype=bool)
    for offset in range(0, ACTIVE_ELEMS, LANES):
        dst_byte_base = offset * BYTES_PER_ELEM
        mask[dst_byte_base : dst_byte_base + LANES] = True
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

    if golden.size != OUTPUT_BUFFER_BYTES:
        print(
            f"[ERROR] Unexpected byte size for this case: got {golden.size}, expected {OUTPUT_BUFFER_BYTES}"
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
