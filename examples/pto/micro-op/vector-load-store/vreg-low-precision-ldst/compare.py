#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/vreg-low-precision-ldst
# family: vector-load-store
# target_ops: pto.vlds, pto.vsts, pto.vldsx2, pto.vstsx2, pto.vsldb, pto.vsstb, pto.vldas, pto.vldus, pto.vstus, pto.vstas

import numpy as np


SEGMENT_NAMES = (
    "vlds/vsts f8e4m3",
    "vlds/vsts f8e5m2",
    "vlds/vsts hif8",
    "vlds/vsts f4e1m2x2",
    "vlds/vsts f4e2m1x2",
    "vldsx2/vstsx2 low",
    "vldsx2/vstsx2 high",
    "vsldb/vsstb hif8",
    "vlds/vsstb/vsldb bf16",
    "vldas/vldus f8e5m2",
    "vstus/vstas f4e2m1x2",
)
VECTOR_BYTES = 256


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=np.uint8)
    output = np.fromfile("v2.bin", dtype=np.uint8)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: golden={golden.shape}, out={output.shape}")
        raise SystemExit(2)
    if not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0])
        segment = idx // VECTOR_BYTES
        name = SEGMENT_NAMES[segment] if segment < len(SEGMENT_NAMES) else f"segment{segment}"
        print(
            f"[ERROR] Low-precision byte roundtrip mismatch: type={name} idx={idx} "
            f"golden=0x{int(golden[idx]):02x} out=0x{int(output[idx]):02x}"
        )
        raise SystemExit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
