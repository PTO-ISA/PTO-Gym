# TileKernels PTO-DSL Migration

This directory tracks the PTO-DSL migration of DeepSeek's TileKernels project.
The migration source is `/Users/zhoubot/TileKernels`; the target is PTO-Gym as
developer-facing examples and validation material.

The migrated kernels are authored in PTO-DSL and validated in two stages:

1. Generate PTO assembly from Python.
2. Run `ptoas` to emit C++ and, in an Ascend environment, compare results with
   the original TileKernels torch/reference behavior.

## Layout

```text
examples/ptodsl/tilekernels/
├── compile_kernels.py          # PTO-DSL -> .pto -> ptoas -> .cpp harness
├── kernel_status.json          # full TileKernels migration checklist
├── manifests/                  # TileKernels GPU correctness shape grids
├── cases/                      # golden/compare assets for migrated cases
└── tilekernels_ptodsl/         # PTO-DSL kernel builders
```

## Quick Start

From the PTO-Gym repository root:

```bash
export PTODSL_ROOT=/Users/zhoubot/github/pto-org/pto-dsl
export PTOAS_BIN=/path/to/ptoas
export PTOAS_FLAGS="--pto-arch a5"
export PYTHONPATH=/path/to/mlir_core:/path/to/PTOAS/python:$PTODSL_ROOT

python3.12 examples/ptodsl/tilekernels/compile_kernels.py --list
python3.12 examples/ptodsl/tilekernels/compile_kernels.py --materialize-validation
```

If `PTOAS_BIN` is not set, the harness tries `ptoas` from `PATH`. The harness
uses `PTOAS_FLAGS` when set and defaults to PTO-Gym's `--pto-arch a5`; it also
adds `--enable-insert-sync` unless that flag is already present. The current
local toolchain uses CPython 3.12 MLIR/PTOAS bindings.

## Validation Assets

`--materialize-validation` writes PTO-Gym case directories under
`.work/ptodsl-tilekernels/validation`. Each generated case contains:

- `kernel.pto`
- `stub.cpp`
- `launch.cpp`
- `main.cpp`
- `golden.py`
- `compare.py`

The PTO-Gym validation scripts accept a generated case root through
`CASES_ROOT`:

```bash
CASES_ROOT=$PWD/.work/ptodsl-tilekernels/validation \
WORK_SPACE=$PWD/.work/vpto-tilekernels \
PTOAS_BIN=/Users/zhoubot/github/pto-org/PTOAS/build-src312/tools/ptoas/ptoas \
PTOAS_FLAGS="--pto-arch a5 --enable-insert-sync" \
COMPILE_ONLY=1 \
examples/pto/scripts/run_host_vpto_validation.sh
```

These generated assets include a host runner plus NumPy golden/compare logic for
deterministic runtime shapes per compiled PTO-DSL variant. Local
`COMPILE_ONLY=1` validation still requires an Ascend/CANN toolchain
(`ASCEND_HOME_PATH`, `bisheng`, and simulator or NPU libraries). Full
correctness is claimed only after `COMPILE_ONLY=0` passes on SIM or NPU.

## Current Kernels

Implemented PTO-DSL builders:

- `moe.normalize_weight`, with `num_topk={1,2,6,7,8,9}` and TileKernels'
  `1e-20` denominator sentinel.
- `moe.mask_indices_by_tp`, scalar int64 expert-index masking/remapping for
  TP-local experts.
- `moe.group_count`, scalar single-block expert count scan for correctness
  validation.
- `moe.aux_fi`, scalar single-block auxiliary frequency indicator scan for
  correctness validation.
- `moe.topk_gate`, PTO sort-based TopK over the TileKernels expert grid using
  `tfillpad_expand`, `sort32`, `mrgsort`, and `gather` as in the PTO-ISA manual
  TopK kernel. PTO-Gym validation currently emits uint32 indices because PTO
  sort exposes uint32 index lanes; TileKernels' public torch API widens to
  int64.
- `moe.topk_sum_and_topk_group_idx`, grouped PTO sort-based TopK for routing
  groups. Each group is padded and sorted to sum its top expert scores, then
  the per-group sums are padded and sorted again to emit uint32 group indices.
  TileKernels' public torch API widens those indices to int64.
- `moe.inplace_unique_group_indices`, correctness-first scalar row scan over
  int64 group ids that preserves first occurrences and writes `-1` for later
  duplicates.
- `transpose.transpose`, for dynamic 2-D bf16/f32 transpose.
- `transpose.batched_transpose`, for dynamic batched bf16/f32 transpose.
- `engram.fused_weight`, for bf16 inputs and f32 output with `hc=4`.
- `engram.engram_hash`, scalar/token-parallel n-gram hash index generation
  with int64 multipliers, vocab modulo, and per-table offsets.
- `engram.engram_gate_fwd`, fixed-hidden Engram gate forward with f32 RMS/dot
  reductions, signed-sqrt sigmoid gating, bf16 output, and saved intermediates.
- `engram.engram_gate_bwd`, correctness-first Engram gate backward producing
  bf16 grad_x/grad_k/grad_v plus f32 per-block `grad_w_partial`.
- `engram.grad_w_reduce`, f32 reduction of persistent partial weight gradients
  with bf16-to-f32 weight conversion and in-place f32 gradient accumulation.
- `mhc.expand_to_mhc_fwd`, for bf16 MHC expansion.
- `mhc.expand_to_mhc_bwd`, for bf16 MHC-axis gradient reduction.
- `mhc.sinkhorn_normalize_fwd` and `mhc.sinkhorn_normalize_bwd`, tile f32
  Sinkhorn ports using `texp`, row/column reductions, row/column expansion, and
  tile arithmetic.
- `mhc.head_compute_mix_fwd` and `mhc.head_compute_mix_bwd`, f32 sigmoid
  head-mix ports using PTO tile exp/reciprocal and a correctness-first
  single-block partial-gradient reducer for backward.
- `mhc.pre_split_mixes_fwd` and `mhc.pre_split_mixes_bwd`, f32 mix-row split
  ports using PTO tile subsets for pre/post/comb branches and a
  correctness-first single-block partial-gradient reducer for backward.
- `mhc.pre_apply_mix_fwd` and `mhc.pre_apply_mix_bwd`, bf16 MHC-lane hidden
  reductions using f32 accumulation and a correctness-first per-token mix
  gradient path for backward.
- `mhc.post_fwd` and `mhc.post_bwd`, bf16 post-layer x plus comb-residual
  combination with correctness-first gradients for x, residual, post mix, and
  comb mix.
- `mhc.pre_norm_fn_fwd`, baseline RMS-normalized residual/FN projection forward
  without optional norm-weight merge.

Shape grids are recorded in
`examples/ptodsl/tilekernels/manifests/tilekernels_gpu_shapes.json`.

`moe.normalize_weight` ports row-wise top-k weight normalization to PTO tile
operations:

- `row_sum` computes the denominator for one token row.
- `tadds` adds TileKernels' `1e-20` denominator sentinel.
- `row_expand` broadcasts the denominator across the row.
- `row_expand_div` computes normalized weights.

The remaining kernels and gap notes are listed in `kernel_status.json` and
documented in `docs/tilekernels-migration.md`.

The scalar MoE count ports intentionally trade throughput for early correctness
coverage: they avoid TileLang atomics by scanning from one PTO block. A later
production pass should replace them with a parallel reduction.
