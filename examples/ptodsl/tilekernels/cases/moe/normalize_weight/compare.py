#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _compare(golden_path: Path, output_path: Path, *, eps: float) -> bool:
    if not golden_path.exists():
        print(f"[ERROR] missing golden: {golden_path}")
        return False
    if not output_path.exists():
        print(f"[ERROR] missing output: {output_path}")
        return False
    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    if golden.shape != output.shape:
        print(f"[ERROR] shape mismatch: {golden.shape} vs {output.shape}")
        return False
    if np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        return True
    diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
    idx = int(np.argmax(diff))
    print(
        f"[ERROR] mismatch: max_diff={float(diff[idx])} idx={idx} "
        f"golden={float(golden[idx])} output={float(output[idx])}"
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("."))
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    ok = True
    ok = _compare(args.dir / "golden_denominator.bin", args.dir / "denominator.bin", eps=args.eps) and ok
    ok = _compare(args.dir / "golden_normalized_weights.bin", args.dir / "normalized_weights.bin", eps=args.eps) and ok
    if not ok:
        raise SystemExit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()

