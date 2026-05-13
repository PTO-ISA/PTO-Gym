#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Debug script to trace trandom computation step by step."""

import numpy as np

TRANDOM_CONST_0 = 0xD2511F53
TRANDOM_CONST_1 = 0xCD9E8D57
TRANDOM_CONST_KEY_ADD_0 = 0x9E3779B9
TRANDOM_CONST_KEY_ADD_1 = 0xBB67AE85

def add_with_128bits_debug(ctr0, ctr1, ctr2, ctr3, value):
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

def trandom_kernel_debug(ctr0, ctr1, ctr2, ctr3, key0_val, key1_val, rounds=10):
    """Philox kernel with detailed logging."""
    lanes = len(ctr0)
    key0 = np.full(lanes, np.uint32(key0_val), dtype=np.uint32)
    key1 = np.full(lanes, np.uint32(key1_val), dtype=np.uint32)
    
    print(f"Initial counters: ctr0[0:5]={ctr0[0:5]}, ctr1[0:5]={ctr1[0:5]}")
    print(f"Initial keys: key0={key0[0]}, key1={key1[0]}")
    
    for round_idx in range(rounds):
        print(f"\n=== Round {round_idx} ===")
        print(f"Before: ctr0[0]={ctr0[0]}, ctr1[0]={ctr1[0]}, ctr2[0]={ctr2[0]}, ctr3[0]={ctr3[0]}")
        print(f"Before: key0={key0[0]}, key1={key1[0]}")
        
        prod0 = ctr0.astype(np.uint64) * np.uint64(TRANDOM_CONST_0)
        prod1 = ctr2.astype(np.uint64) * np.uint64(TRANDOM_CONST_1)
        
        L0 = prod0.astype(np.uint32)
        H0 = (prod0 >> 32).astype(np.uint32)
        L1 = prod1.astype(np.uint32)
        H1 = (prod1 >> 32).astype(np.uint32)
        
        print(f"prod0[0]={prod0[0]}, L0[0]={L0[0]}, H0[0]={H0[0]}")
        print(f"prod1[0]={prod1[0]}, L1[0]={L1[0]}, H1[0]={H1[0]}")
        
        ctr0 = (H1 ^ ctr1) ^ key0
        ctr2 = (H0 ^ ctr3) ^ key1
        
        print(f"ctr0[0] = (H1[0] ^ ctr1[0]) ^ key0[0] = ({H1[0]} ^ {ctr1[0]}) ^ {key0[0]} = {ctr0[0]}")
        print(f"ctr2[0] = (H0[0] ^ ctr3[0]) ^ key1[0] = ({H0[0]} ^ {ctr3[0]}) ^ {key1[0]} = {ctr2[0]}")
        
        key0 = (key0.astype(np.uint32) + np.uint32(TRANDOM_CONST_KEY_ADD_0)) & np.uint32(0xFFFFFFFF)
        key1 = (key1.astype(np.uint32) + np.uint32(TRANDOM_CONST_KEY_ADD_1)) & np.uint32(0xFFFFFFFF)
        
        print(f"key0={key0[0]}, key1={key1[0]} (after update)")
        
        ctr1 = L1
        ctr3 = L0
        
        print(f"After: ctr0[0]={ctr0[0]}, ctr1[0]={ctr1[0]}, ctr2[0]={ctr2[0]}, ctr3[0]={ctr3[0]}")
    
    return ctr0, ctr1, ctr2, ctr3

key = np.array([-792737938, 2139558336], dtype=np.int32)
counter = np.array([-1759534764, -1881674653, 640338625, 1381573024], dtype=np.int32)

key_uint = key.view(np.uint32)
counter_uint = counter.view(np.uint32)

lanes = 64
ctr0 = np.full(lanes, counter_uint[0], dtype=np.uint32)
ctr1 = np.full(lanes, counter_uint[1], dtype=np.uint32)
ctr2 = np.full(lanes, counter_uint[2], dtype=np.uint32)
ctr3 = np.full(lanes, counter_uint[3], dtype=np.uint32)

print("=== Initial counter values ===")
print(f"ctr0[0]={ctr0[0]}, ctr1[0]={ctr1[0]}, ctr2[0]={ctr2[0]}, ctr3[0]={ctr3[0]}")

inc_idx = np.arange(lanes, dtype=np.uint32)
ctr0, ctr1, ctr2, ctr3 = add_with_128bits_debug(ctr0, ctr1, ctr2, ctr3, inc_idx)

print("\n=== After adding index ===")
print(f"ctr0[0:5]={ctr0[0:5]}")
print(f"ctr1[0:5]={ctr1[0:5]}")
print(f"ctr2[0:5]={ctr2[0:5]}")
print(f"ctr3[0:5]={ctr3[0:5]}")

result = trandom_kernel_debug(ctr0.copy(), ctr1.copy(), ctr2.copy(), ctr3.copy(), 
                               key_uint[0], key_uint[1], rounds=10)

print("\n=== Final result ===")
print(f"ctr0[0:5]={result[0][0:5]}")
print(f"ctr1[0:5]={result[1][0:5]}")
print(f"ctr2[0:5]={result[2][0:5]}")
print(f"ctr3[0:5]={result[3][0:5]}")