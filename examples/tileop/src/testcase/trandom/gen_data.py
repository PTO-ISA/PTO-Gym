#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input data and golden output for trandom test cases.

Implements the Philox-based TRandom algorithm in pure Python/NumPy
to generate reference golden data for comparison with NPU output.

Flow:
  - First run (no output.bin): generate key/counter inputs only
  - Second run (with output.bin): read saved key/counter, compute golden
"""

import os
import numpy as np
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

TRANDOM_ONCE_REPEAT = 4
TRANDOM_CONST_0 = 0xD2511F53
TRANDOM_CONST_1 = 0xCD9E8D57
TRANDOM_CONST_KEY_ADD_0 = 0x9E3779B9
TRANDOM_CONST_KEY_ADD_1 = 0xBB67AE85


def add_with_128bits(ctr0, ctr1, ctr2, ctr3, value):
    """Simulate 128-bit addition with carry propagation."""
    ctr0_new = ctr0.astype(np.uint64) + value.astype(np.uint64)
    carry0 = (ctr0_new > 0xFFFFFFFF).astype(np.uint32)
    ctr0_new = ctr0_new.astype(np.uint32)
    
    ctr1_new = ctr1.astype(np.uint64) + carry0.astype(np.uint64)
    carry1 = (ctr1_new > 0xFFFFFFFF).astype(np.uint32)
    ctr1_new = ctr1_new.astype(np.uint32)
    
    ctr2_new = ctr2.astype(np.uint64) + carry1.astype(np.uint64)
    carry2 = (ctr2_new > 0xFFFFFFFF).astype(np.uint32)
    ctr2_new = ctr2_new.astype(np.uint32)
    
    ctr3_new = ctr3.astype(np.uint64) + carry2.astype(np.uint64)
    ctr3_new = ctr3_new.astype(np.uint32)
    
    return ctr0_new, ctr1_new, ctr2_new, ctr3_new


def trandom_kernel(ctr0, ctr1, ctr2, ctr3, key0_val, key1_val, rounds=10):
    """Philox-based random number generation kernel.

    Uses unsigned multiply to match C++ TRandomKernel (RegTensor<uint32_t>, vmull.v64u32).
    """
    key0 = np.full(len(ctr0), key0_val, dtype=np.uint32)
    key1 = np.full(len(ctr0), key1_val, dtype=np.uint32)

    for _ in range(rounds):
        prod0 = ctr0.astype(np.uint64) * np.uint64(TRANDOM_CONST_0)
        prod1 = ctr2.astype(np.uint64) * np.uint64(TRANDOM_CONST_1)

        L0 = prod0.astype(np.uint32)
        H0 = (prod0 >> 32).astype(np.uint32)
        L1 = prod1.astype(np.uint32)
        H1 = (prod1 >> 32).astype(np.uint32)

        ctr0 = (H1 ^ ctr1) ^ key0
        ctr2 = (H0 ^ ctr3) ^ key1
        ctr1 = L1
        ctr3 = L0

        key0 = (key0 + TRANDOM_CONST_KEY_ADD_0) & np.uint32(0xFFFFFFFF)
        key1 = (key1 + TRANDOM_CONST_KEY_ADD_1) & np.uint32(0xFFFFFFFF)

    return ctr0, ctr1, ctr2, ctr3


def interleave_values(ctr0, ctr1, ctr2, ctr3):
    """Simulate vintlv: interleave values to reorder random numbers.
    
    vintlv semantics (N=64, half=32):
    - low[2*i] = src0[i], low[2*i+1] = src1[i] for i in 0..31 (interleave first half)
    - high[2*i] = src0[i+32], high[2*i+1] = src1[i+32] for i in 0..31 (interleave second half)
    
    TRandom uses:
    1. vintlv(tmpL0, tmpH0, ctr0, ctr2)
    2. vintlv(tmpL1, tmpH1, ctr1, ctr3)
    3. vintlv(ctr0, ctr1, tmpL0, tmpL1)
    4. vintlv(ctr2, ctr3, tmpH0, tmpH1)
    """
    n = len(ctr0)
    half = n // 2
    
    tmpL0 = np.empty(n, dtype=np.uint32)
    tmpH0 = np.empty(n, dtype=np.uint32)
    tmpL1 = np.empty(n, dtype=np.uint32)
    tmpH1 = np.empty(n, dtype=np.uint32)
    
    for i in range(half):
        tmpL0[2*i] = ctr0[i]
        tmpL0[2*i+1] = ctr2[i]
        tmpH0[2*i] = ctr0[i + half]
        tmpH0[2*i+1] = ctr2[i + half]
        
        tmpL1[2*i] = ctr1[i]
        tmpL1[2*i+1] = ctr3[i]
        tmpH1[2*i] = ctr1[i + half]
        tmpH1[2*i+1] = ctr3[i + half]
    
    result0 = np.empty(n, dtype=np.uint32)
    result1 = np.empty(n, dtype=np.uint32)
    result2 = np.empty(n, dtype=np.uint32)
    result3 = np.empty(n, dtype=np.uint32)
    
    for i in range(half):
        result0[2*i] = tmpL0[i]
        result0[2*i+1] = tmpL1[i]
        result1[2*i] = tmpL0[i + half]
        result1[2*i+1] = tmpL1[i + half]
        
        result2[2*i] = tmpH0[i]
        result2[2*i+1] = tmpH1[i]
        result3[2*i] = tmpH0[i + half]
        result3[2*i+1] = tmpH1[i + half]
    
    return result0, result1, result2, result3


def trandom_generate(key, counter, valid_rows, valid_cols, dtype=np.int32, rounds=10):
    """Generate random numbers using TRandom algorithm.
    
    Args:
        key: 2-element array (key0, key1) - scalar values, broadcast to all lanes
        counter: 4-element array (counter0-3) - 128-bit counter base value
        valid_rows: number of rows to generate
        valid_cols: number of columns to generate
        dtype: output dtype (int32 or uint32)
        rounds: number of Philox rounds (7 or 10)
    
    Returns:
        output: (valid_rows, valid_cols) array of random numbers
    """
    lanes = 64
    n_loop = (valid_cols + TRANDOM_ONCE_REPEAT * lanes - 1) // (TRANDOM_ONCE_REPEAT * lanes)
    
    output = np.zeros((valid_rows, valid_cols), dtype=np.uint32)
    
    key0_val = np.uint32(key[0])
    key1_val = np.uint32(key[1])
    
    ctr0 = np.full(lanes, np.uint32(counter[0]), dtype=np.uint32)
    ctr1 = np.full(lanes, np.uint32(counter[1]), dtype=np.uint32)
    ctr2 = np.full(lanes, np.uint32(counter[2]), dtype=np.uint32)
    ctr3 = np.full(lanes, np.uint32(counter[3]), dtype=np.uint32)
    
    inc_idx = np.arange(lanes, dtype=np.uint32)
    ctr0, ctr1, ctr2, ctr3 = add_with_128bits(ctr0, ctr1, ctr2, ctr3, inc_idx)
    
    for i in range(valid_rows):
        s_reg = valid_cols
        counter_add_val = lanes
        
        for j in range(n_loop):
            tmp_ctr0 = ctr0.copy()
            tmp_ctr1 = ctr1.copy()
            tmp_ctr2 = ctr2.copy()
            tmp_ctr3 = ctr3.copy()
            
            tmp_ctr0, tmp_ctr1, tmp_ctr2, tmp_ctr3 = trandom_kernel(
                tmp_ctr0, tmp_ctr1, tmp_ctr2, tmp_ctr3, key0_val, key1_val, rounds=rounds
            )
            
            # Apply interleave to match vintlv semantics in trandom_template.py
            # This produces element-wise interleaved order: [ctr0[0], ctr1[0], ctr2[0], ctr3[0], ...]
            tmp_ctr0, tmp_ctr1, tmp_ctr2, tmp_ctr3 = interleave_values(
                tmp_ctr0, tmp_ctr1, tmp_ctr2, tmp_ctr3
            )
            
            for k in range(TRANDOM_ONCE_REPEAT):
                start_col = TRANDOM_ONCE_REPEAT * j * lanes + k * lanes
                end_col = min(start_col + lanes, valid_cols)
                num_valid = end_col - start_col
                
                if num_valid > 0:
                    vals = [tmp_ctr0, tmp_ctr1, tmp_ctr2, tmp_ctr3][k]
                    output[i, start_col:end_col] = vals[:num_valid]
            
            if s_reg >= TRANDOM_ONCE_REPEAT * lanes:
                s_reg = s_reg - TRANDOM_ONCE_REPEAT * lanes
            else:
                s_reg = 0
            
            counter_add_val = lanes if j != n_loop - 1 else ((valid_cols - 1) % lanes + 1)
            v_ele_stride = np.full(lanes, np.uint32(counter_add_val), dtype=np.uint32)
            ctr0, ctr1, ctr2, ctr3 = add_with_128bits(ctr0, ctr1, ctr2, ctr3, v_ele_stride)
    
    return output.view(dtype)


validate_cases(CASES)

for case in CASES:
    case_dir = case["name"]
    key_file = os.path.join(case_dir, "key.bin")
    counter_file = os.path.join(case_dir, "counter.bin")
    output_file = os.path.join(case_dir, "output.bin")
    
    dtype = case["dtype"]
    valid_rows, valid_cols = case["valid_shape"]
    rounds = case.get("rounds", 10)
    
    if os.path.exists(key_file) and os.path.exists(counter_file):
        key = np.fromfile(key_file, dtype=dtype)
        counter = np.fromfile(counter_file, dtype=dtype)
        print(f"[INFO] gen_data: {case['name']} loaded existing key/counter")
    else:
        setup_case_rng(case)
        value_max = np.iinfo(dtype).max
        value_min = np.iinfo(dtype).min
        key = np.random.randint(value_min, value_max + 1, size=2, dtype=dtype)
        counter = np.random.randint(value_min, value_max + 1, size=4, dtype=dtype)
        print(f"[INFO] gen_data: {case['name']} generated new key={key.tolist()} counter={counter.tolist()}")
    
    if os.path.exists(output_file):
        golden = trandom_generate(key.view(np.uint32), counter.view(np.uint32),
                                   valid_rows, valid_cols, dtype=dtype, rounds=rounds)
        save_case_data(case["name"], {"key": key, "counter": counter, "golden": golden})
        print(f"[INFO] gen_data: {case['name']} generated golden shape={case['shape']}")
    else:
        save_case_data(case["name"], {"key": key, "counter": counter})
        print(f"[INFO] gen_data: {case['name']} saved inputs (waiting for output)")