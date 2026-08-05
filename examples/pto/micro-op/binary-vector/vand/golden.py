#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Merged vand test case.

import argparse
from pathlib import Path
import numpy as np

ROWS = 32
COLS = 32
SEED = 19
LOGICAL_ELEMS = 1000
OUT_SENTINEL = np.float32(-123.25)

def f32_to_bf16_bits(v):
    w=v.astype(np.float32,copy=False).view(np.uint32)
    r=np.uint32(0x7FFF)+((w>>16)&np.uint32(1))
    return ((w+r)>>16).astype(np.uint16)
def bf16_bits_to_f32(b):
    return (b.astype(np.uint32)<<16).view(np.float32)


# ---- f32 ----
def gen_f32(out, rng):
    v1=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
    v2=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
    g=np.bitwise_and(v1,v2)
    v3=np.zeros((ROWS,COLS),dtype=np.uint16)
    v1.reshape(-1).tofile(out/"v1.bin")
    v2.reshape(-1).tofile(out/"v2.bin")
    v3.reshape(-1).tofile(out/"v3.bin")
    g.reshape(-1).tofile(out/"golden_v3.bin")

# ---- mask_edge ----
def gen_mask_edge(out, rng):
    v1=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
    v2=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
    g=np.bitwise_and(v1,v2)
    v3=np.zeros((ROWS,COLS),dtype=np.uint16)
    v1.reshape(-1).tofile(out/"v1_mask_edge.bin")
    v2.reshape(-1).tofile(out/"v2_mask_edge.bin")
    v3.reshape(-1).tofile(out/"v3_mask_edge.bin")
    g.reshape(-1).tofile(out/"golden_v3_mask_edge.bin")

# ---- lowp ----
def gen_lowp(out, rng):
    f8_v1=rng.integers(0,256,size=256,dtype=np.uint8)
    f8_v2=rng.integers(0,256,size=256,dtype=np.uint8)
    hif8_v1=rng.integers(0,256,size=256,dtype=np.uint8)
    hif8_v2=rng.integers(0,256,size=256,dtype=np.uint8)
    zeros=np.zeros(256,dtype=np.uint8)
    f8_v1.tofile(out/"v1_f8.bin")
    f8_v2.tofile(out/"v2_f8.bin")
    zeros.tofile(out/"v3_f8_and.bin")
    zeros.tofile(out/"v4_f8_xor.bin")
    zeros.tofile(out/"v5_f8_or.bin")
    hif8_v1.tofile(out/"v1_hif8.bin")
    hif8_v2.tofile(out/"v2_hif8.bin")
    zeros.tofile(out/"v3_hif8_and.bin")
    zeros.tofile(out/"v4_hif8_xor.bin")
    zeros.tofile(out/"v5_hif8_or.bin")
    np.bitwise_and(f8_v1,f8_v2).tofile(out/"golden_v3_f8_and.bin")
    np.bitwise_xor(f8_v1,f8_v2).tofile(out/"golden_v4_f8_xor.bin")
    np.bitwise_or(f8_v1,f8_v2).tofile(out/"golden_v5_f8_or.bin")
    np.bitwise_and(hif8_v1,hif8_v2).tofile(out/"golden_v3_hif8_and.bin")
    np.bitwise_xor(hif8_v1,hif8_v2).tofile(out/"golden_v4_hif8_xor.bin")
    np.bitwise_or(hif8_v1,hif8_v2).tofile(out/"golden_v5_hif8_or.bin")

GENERATORS = [
    gen_f32,
    gen_mask_edge,
    gen_lowp,
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=Path("."))
    p.add_argument("--seed",type=int,default=SEED)
    a=p.parse_args()
    rng=np.random.default_rng(a.seed)
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    for gen in GENERATORS:
        gen(out,rng)

if __name__=="__main__":
    main()
