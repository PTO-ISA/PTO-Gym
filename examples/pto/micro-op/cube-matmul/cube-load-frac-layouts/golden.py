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


def nchw_to_nc1hwc0(nchw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    n, c, h, w = nchw_tensor.shape
    c1 = (c + c0 - 1) // c0
    padded = np.pad(nchw_tensor, ((0, 0), (0, c1 * c0 - c), (0, 0), (0, 0)))
    return np.transpose(padded.reshape(n, c1, c0, h, w), (0, 1, 3, 4, 2))


def nchw_to_c1hw_n16_16_c0(nchw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    n, c, h, w = nchw_tensor.shape
    n_pad = ((n + 15) // 16) * 16
    c_pad = ((c + c0 - 1) // c0) * c0
    c1 = c_pad // c0
    padded = np.pad(nchw_tensor, ((0, n_pad - n), (0, c_pad - c), (0, 0), (0, 0)))
    nc1c0hw = padded.reshape(n_pad, c1, c0, h, w)
    n16 = nc1c0hw.reshape(n_pad // 16, 16, c1, c0, h, w)
    return np.transpose(n16, (2, 4, 5, 0, 1, 3)).reshape(c1 * h * w, n_pad // 16, 16, c0)


def ncdhw_to_ndc1hwc0(ncdhw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    n, c, d, h, w = ncdhw_tensor.shape
    c1 = (c + c0 - 1) // c0
    padded = np.pad(ncdhw_tensor, ((0, 0), (0, c1 * c0 - c), (0, 0), (0, 0), (0, 0)))
    nc1c0dhw = padded.reshape(n, c1, c0, d, h, w)
    return np.transpose(nc1c0dhw, (0, 3, 1, 4, 5, 2))


def ncdhw_to_c1dhw_n16_16_c0(ncdhw_tensor: np.ndarray, c0: int = 16) -> np.ndarray:
    n, c, d, h, w = ncdhw_tensor.shape
    n_pad = ((n + 15) // 16) * 16
    c_pad = ((c + c0 - 1) // c0) * c0
    c1 = c_pad // c0
    padded = np.pad(ncdhw_tensor, ((0, n_pad - n), (0, c_pad - c), (0, 0), (0, 0), (0, 0)))
    nc1c0dhw = padded.reshape(n_pad, c1, c0, d, h, w)
    n16 = nc1c0dhw.reshape(n_pad // 16, 16, c1, c0, d, h, w)
    return np.transpose(n16, (2, 4, 5, 6, 0, 1, 3)).reshape(c1 * d * h * w, n_pad // 16, 16, c0)


def write(path: Path, array: np.ndarray) -> None:
    array.reshape(-1).tofile(path)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lhs_nd2nz_case1 = (np.arange(40 * 50, dtype=np.float16).reshape(40, 50) * np.float16(0.5) +
                       np.float16(17)).astype(np.float16)
    nd2nz_case1 = (np.arange(50 * 60, dtype=np.float16).reshape(50, 60) * np.float16(0.25) +
                   np.float16(3)).astype(np.float16)
    lhs = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) + 2001).astype(np.float16)
    dn2nz = (np.arange(16 * 16, dtype=np.float16).reshape(16, 16) + 301).astype(np.float16)
    nchw = (np.arange(1 * 10 * 1 * 16, dtype=np.float16).reshape(1, 10, 1, 16) + 601).astype(np.float16)
    nchw_fz4d = (np.arange(5 * 16, dtype=np.float16).reshape(5, 16, 1, 1) + 901).astype(np.float16)
    ncdhw = (np.arange(1 * 7 * 1 * 1 * 16, dtype=np.float16).reshape(1, 7, 1, 1, 16) + 1201).astype(np.float16)
    ncdhw_fz3d = (np.arange(3 * 16, dtype=np.float16).reshape(3, 16, 1, 1, 1) + 1501).astype(np.float16)

    lhs_nd2nz_case1_f32 = lhs_nd2nz_case1.astype(np.float32)
    lhs_f32 = lhs.astype(np.float32)
    golden_nd2nz = lhs_nd2nz_case1_f32 @ nd2nz_case1.astype(np.float32)
    golden_dn2nz = lhs_f32 @ dn2nz.astype(np.float32)
    golden_nchw_nc1hwc0 = lhs_f32 @ nchw_to_nc1hwc0(nchw).reshape(16, 16).astype(np.float32)
    golden_nchw_fz4d = lhs_f32 @ nchw_to_c1hw_n16_16_c0(nchw_fz4d).reshape(16, 16).astype(np.float32)
    golden_ncdhw_ndc1hwc0 = lhs_f32 @ ncdhw_to_ndc1hwc0(ncdhw).reshape(16, 16).astype(np.float32)
    golden_ncdhw_fz3d = lhs_f32 @ ncdhw_to_c1dhw_n16_16_c0(ncdhw_fz3d).reshape(16, 16).astype(np.float32)

    zeros_nd2nz = np.zeros((40, 60), dtype=np.float32)
    zeros = np.zeros((16, 16), dtype=np.float32)

    write(output_dir / "lhs_nd2nz_case1.bin", lhs_nd2nz_case1)
    write(output_dir / "src_nd2nz_case1.bin", nd2nz_case1)
    write(output_dir / "identity.bin", lhs)
    write(output_dir / "src_dn2nz.bin", dn2nz)
    write(output_dir / "src_nchw_nc1hwc0.bin", nchw)
    write(output_dir / "src_nchw_fz4d.bin", nchw_fz4d)
    write(output_dir / "src_ncdhw_ndc1hwc0.bin", ncdhw)
    write(output_dir / "src_ncdhw_fz3d.bin", ncdhw_fz3d)

    write(output_dir / "out_nd2nz_case1.bin", zeros_nd2nz)
    write(output_dir / "out_dn2nz.bin", zeros)
    write(output_dir / "out_nchw_nc1hwc0.bin", zeros)
    write(output_dir / "out_nchw_fz4d.bin", zeros)
    write(output_dir / "out_ncdhw_ndc1hwc0.bin", zeros)
    write(output_dir / "out_ncdhw_fz3d.bin", zeros)

    write(output_dir / "golden_nd2nz_case1.bin", golden_nd2nz)
    write(output_dir / "golden_dn2nz.bin", golden_dn2nz)
    write(output_dir / "golden_nchw_nc1hwc0.bin", golden_nchw_nc1hwc0)
    write(output_dir / "golden_nchw_fz4d.bin", golden_nchw_fz4d)
    write(output_dir / "golden_ncdhw_ndc1hwc0.bin", golden_ncdhw_ndc1hwc0)
    write(output_dir / "golden_ncdhw_fz3d.bin", golden_ncdhw_fz3d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
