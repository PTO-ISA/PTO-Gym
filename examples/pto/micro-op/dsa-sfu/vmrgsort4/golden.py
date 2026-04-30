#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import struct
from pathlib import Path


PAIR_FMT = "fI"


def write_pairs(path: Path, pairs) -> None:
    with path.open("wb") as f:
        for score, index in pairs:
            f.write(struct.pack(PAIR_FMT, score, index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    src = [(9.0, 90), (7.0, 70), (8.0, 80), (6.0, 60)]
    golden = [(9.0, 90), (8.0, 80), (7.0, 70), (6.0, 60)]

    write_pairs(out / "v1.bin", src)
    write_pairs(out / "v2.bin", [(0.0, 0)] * 4)
    write_pairs(out / "golden_v2.bin", golden)


if __name__ == "__main__":
    main()
