#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: kernels/flash-attention
# family: kernels
# target_ops: pto.mte_gm_l1_frac, pto.mte_l1_l0a, pto.mte_l1_l0b, pto.mad,
#   pto.mte_l0c_ub, pto.mte_gm_ub, pto.mte_ub_gm, pto.vlds, pto.vcmax,
#   pto.vdup, pto.vmax, pto.vexpdif, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv,
#   pto.vsts, pto.sync.set, pto.sync.wait
# scenarios: flash-attention, cube-qk, tiled-online-softmax, q32-k32-d8

import os
import sys

import numpy as np


def main():
    if not os.path.exists("v4.bin"):
        print("[ERROR] Output missing: v4.bin")
        sys.exit(2)
    if not os.path.exists("golden_v4.bin"):
        print("[ERROR] Golden missing: golden_v4.bin")
        sys.exit(2)
    output = np.fromfile("v4.bin", dtype=np.float32)
    golden = np.fromfile("golden_v4.bin", dtype=np.float32)
    if output.shape != golden.shape:
        print(f"[ERROR] Shape mismatch: golden={golden.shape}, out={output.shape}")
        sys.exit(2)
    if not np.allclose(golden, output, atol=2e-3, rtol=2e-3, equal_nan=True):
        diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        idx = int(np.argmax(diff))
        print(
            f"[ERROR] Mismatch: max diff={float(diff[idx])} at idx={idx} "
            f"(golden={float(golden[idx])}, out={float(output[idx])})"
        )
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
