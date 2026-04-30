#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/predicate-load-store/psts-norm-plds-ds
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
