#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/vector-load-store/dma-copy-rearrange
# family: micro-op/vector-load-store
# target_ops: pto.copy_gm_to_ubuf, pto.mte_ub_ub, pto.copy_ubuf_to_gm
# scenarios: i16, ub-rearrange, permute-4x16-rows

import argparse
from pathlib import Path

import numpy as np


ROWS = 4
COLS = 16


def generate(output_dir: Path) -> None:
    v1 = np.arange(ROWS * COLS, dtype=np.int16).reshape(ROWS, COLS)
    v2 = np.zeros((ROWS, COLS), dtype=np.int16)
    golden_v2 = v1[[2, 0, 3, 1], :].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    v1.reshape(-1).tofile(output_dir / "v1.bin")
    v2.reshape(-1).tofile(output_dir / "v2.bin")
    golden_v2.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
