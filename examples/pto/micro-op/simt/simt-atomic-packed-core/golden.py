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

ELEMS = 16


def pack_u16_pair(lo: int, hi: int) -> np.uint32:
    return np.uint32((hi << 16) | lo)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h2 = np.zeros(ELEMS, dtype=np.uint32)
    b2 = np.zeros(ELEMS, dtype=np.uint32)
    h2[0] = pack_u16_pair(0x3C00, 0x3C00)  # f16x2: (1.0, 1.0)
    h2[1] = pack_u16_pair(0x3C00, 0x4000)  # f16x2: (1.0, 2.0)
    b2[0] = pack_u16_pair(0x3F80, 0x4040)  # bf16x2: (1.0, 3.0)
    b2[1] = pack_u16_pair(0x4040, 0x4040)  # bf16x2: (3.0, 3.0)

    golden_h2 = h2.copy()
    golden_b2 = b2.copy()
    golden_h2[0] = pack_u16_pair(0x4000, 0x4000)  # CAS -> (2.0, 2.0)
    golden_h2[1] = pack_u16_pair(0x4000, 0x4200)  # ADD -> (2.0, 3.0)
    golden_b2[0] = pack_u16_pair(0x4000, 0x4000)  # EXCH -> (2.0, 2.0)
    golden_b2[1] = pack_u16_pair(0x3F80, 0x3F80)  # MIN -> (1.0, 1.0)

    h2.tofile(output_dir / "v1.bin")
    b2.tofile(output_dir / "v2.bin")
    golden_h2.tofile(output_dir / "golden_v1.bin")
    golden_b2.tofile(output_dir / "golden_v2.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
