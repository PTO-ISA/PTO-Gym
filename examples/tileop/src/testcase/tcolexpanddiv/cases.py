# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in the compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcolexpanddiv ST test cases.
Matches PTO-ISA testcase definitions in /home/zhoushaofan/code/pto-isa/tests/npu/a5/src/st/testcase/tcolexpanddiv/

TCOLEXPANDDIV: column-wise broadcast divide - dst[i,j] = src0[i,j] / src1[0,j]
  - src0_shape: (src0_row, cols)   - dividend input tile
  - src1_shape: (1, cols)          - divisor input tile (single row, broadcast)
  - dst_shape: (dst_row, cols)     - output tile
  - valid_shape: (valid_row, valid_col) - effective computation region

Case naming: {dtype}_{src0_row}_{src0_col}_{src1_row}_{src1_col}
"""

import numpy as np

CASES = [
    {
        "name": "fp32_32_64_1_64",
        "dtype": np.float32,
        "src0_shape": (32, 64),
        "src1_shape": (1, 64),
        "shape": (32, 64),
        "valid_shape": (32, 64),
        "eps": 1e-6,
    },
    {
        "name": "fp32_8_32_1_32",
        "dtype": np.float32,
        "src0_shape": (8, 32),
        "src1_shape": (1, 32),
        "shape": (8, 32),
        "valid_shape": (8, 32),
        "eps": 1e-6,
    },
    {
        "name": "fp16_16_64_1_64",
        "dtype": np.float16,
        "src0_shape": (16, 64),
        "src1_shape": (1, 64),
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-3,
    },
    {
        "name": "fp16_4_128_1_128",
        "dtype": np.float16,
        "src0_shape": (4, 128),
        "src1_shape": (1, 128),
        "shape": (4, 128),
        "valid_shape": (4, 128),
        "eps": 1e-3,
    },
    {
        "name": "int32_16_32_1_32",
        "dtype": np.int32,
        "src0_shape": (16, 32),
        "src1_shape": (1, 32),
        "shape": (16, 32),
        "valid_shape": (16, 32),
        "eps": 0,
    },
    {
        "name": "int16_16_64_1_64",
        "dtype": np.int16,
        "src0_shape": (16, 64),
        "src1_shape": (1, 64),
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 0,
    },
    {
        "name": "fp32_40_32_1_32",
        "dtype": np.float32,
        "src0_shape": (40, 32),
        "src1_shape": (1, 32),
        "shape": (40, 32),
        "valid_shape": (40, 32),
        "eps": 1e-6,
    },
    {
        "name": "fp16_16_128_1_128",
        "dtype": np.float16,
        "src0_shape": (16, 128),
        "src1_shape": (1, 128),
        "shape": (16, 128),
        "valid_shape": (16, 128),
        "eps": 1e-3,
    },
    {
        "name": "fp32_20_64_1_64",
        "dtype": np.float32,
        "src0_shape": (20, 64),
        "src1_shape": (1, 64),
        "shape": (20, 64),
        "valid_shape": (20, 64),
        "eps": 1e-6,
    },
]