# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmins ST test cases."""

import numpy as np

CASES = [
    {"name": "f32_32x64", "dtype": np.float32, "shape": (32, 64), "valid_shape": (32, 64), "eps": 1e-6},
    {"name": "f16_63x64", "dtype": np.float16, "shape": (63, 64), "valid_shape": (63, 64), "eps": 1e-3},
    {"name": "i32_31x128", "dtype": np.int32, "shape": (31, 128), "valid_shape": (31, 128), "eps": 0},
    {"name": "i16_15x192", "dtype": np.int16, "shape": (15, 192), "valid_shape": (15, 192), "eps": 0},
    {"name": "f32_7x448", "dtype": np.float32, "shape": (7, 448), "valid_shape": (7, 448), "eps": 1e-6},
    {"name": "f32_256x16", "dtype": np.float32, "shape": (256, 16), "valid_shape": (256, 16), "eps": 1e-6},
]
