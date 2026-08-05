#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Merged vshr test case.

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
    v2=rng.integers(0,16,size=(ROWS,COLS),dtype=np.uint16)
    g=np.right_shift(v1.astype(np.uint16),v2.astype(np.uint16))
    v3=np.zeros((ROWS,COLS),dtype=np.uint16)
    v1.reshape(-1).tofile(out/"v1.bin")
    v2.reshape(-1).tofile(out/"v2.bin")
    v3.reshape(-1).tofile(out/"v3.bin")
    g.reshape(-1).tofile(out/"golden_v3.bin")

# ---- i16_signed ----
def gen_i16_signed(out, rng):
    v1=rng.integers(-1000,1001,size=(ROWS,COLS),dtype=np.int16)
    v2=rng.integers(0,16,size=(ROWS,COLS),dtype=np.int16)
    g=np.right_shift(v1.astype(np.int32),v2.astype(np.int32)).astype(np.int16)
    v3=np.zeros((ROWS,COLS),dtype=np.int16)
    v1.reshape(-1).tofile(out/"v1_i16_signed.bin")
    v2.reshape(-1).tofile(out/"v2_i16_signed.bin")
    v3.reshape(-1).tofile(out/"v3_i16_signed.bin")
    g.reshape(-1).tofile(out/"golden_v3_i16_signed.bin")

# ---- shift_boundary ----
def gen_shift_boundary(out, rng):
    elems=ROWS*COLS
    lhs_pat=np.array([0,1,15,16,255,256,4095,4096,32767,32768,65535,0],dtype=np.uint16)
    rhs_pat=np.array([0,1,1,1,2,2,3,3,4,5,6,15],dtype=np.uint16)
    reps=elems//lhs_pat.size
    v1=np.resize(lhs_pat,elems)
    v2=np.resize(rhs_pat,elems)
    g=np.right_shift(v1.astype(np.uint16),v2.astype(np.uint16))
    v3=np.zeros(elems,dtype=np.uint16)
    v1.tofile(out/"v1_shift_boundary.bin")
    v2.tofile(out/"v2_shift_boundary.bin")
    v3.tofile(out/"v3_shift_boundary.bin")
    g.tofile(out/"golden_v3_shift_boundary.bin")

GENERATORS = [
    gen_f32,
    gen_i16_signed,
    gen_shift_boundary,
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
