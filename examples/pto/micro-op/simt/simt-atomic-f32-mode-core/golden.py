#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

F32_ELEMS = 32
PACKED_ELEMS = 16


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(F32_ELEMS, -1.0, dtype=np.float32)
    golden_v1 = np.full(F32_ELEMS, -1.0, dtype=np.float32)
    v2 = np.full(PACKED_ELEMS, 0xBC00, dtype=np.uint16)
    v3 = np.full(PACKED_ELEMS, 0xBF80, dtype=np.uint16)
    golden_v2 = v2.copy()
    golden_v3 = v3.copy()
    v1[:4] = np.full(4, 10.0, dtype=np.float32)
    golden_v1[:4] = np.full(4, 15.0, dtype=np.float32)
    golden_v1[16:20] = np.full(4, 10.0, dtype=np.float32)
    v2[:2] = np.array([0x3C00, 0x4000], dtype=np.uint16)  # f16: 1.0, 2.0
    v3[:2] = np.array([0x3F80, 0x4040], dtype=np.uint16)  # bf16: 1.0, 3.0
    golden_v2[:2] = np.array([0x4000, 0x4200], dtype=np.uint16)  # 2.0, 3.0
    golden_v3[:2] = np.array([0x4000, 0x4000], dtype=np.uint16)  # 2.0, 2.0
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    v3.tofile(output_dir / "v3.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")
    golden_v3.tofile(output_dir / "golden_v3.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
