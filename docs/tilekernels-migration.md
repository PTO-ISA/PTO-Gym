# TileKernels PTO-DSL Migration

This document tracks the migration from DeepSeek TileKernels to PTO-Gym.

Source repository:

- `/Users/zhoubot/TileKernels`
- upstream: `https://github.com/deepseek-ai/TileKernels`

Target repository:

- `PTO-ISA/PTO-Gym`
- local checkout: `/Users/zhoubot/github/pto-org/PTO-Gym`

## Acceptance Gates

Each kernel is complete only when all gates pass:

- PTO-DSL source exists under `examples/ptodsl/tilekernels/tilekernels_ptodsl`.
- The compile harness writes `.pto` and `ptoas` emits nonempty C++.
- PTO-Gym validation scripts can compile the generated case assets with
  `COMPILE_ONLY=1`.
- Correctness matches the TileKernels torch/reference behavior on supported NPU
  or simulator configurations.
- Shape, dtype, layout, and unsupported-feature notes are documented.

## Current Implementation

The migration scaffold is in `examples/ptodsl/tilekernels`. The harness now
generates PTO-Gym validation case directories under
`.work/ptodsl-tilekernels/validation` when run with
`--materialize-validation`.

Implemented kernels:

| Kernel | Family | PTO-DSL | ptoas C++ | Validation Assets | SIM/NPU Correct | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `normalize_weight` | MoE | yes | yes | generated | pending | Uses `row_sum`, `tadds` for TileKernels' `1e-20` denominator sentinel, `row_expand`, and `row_expand_div`; configs cover `num_topk` 1, 2, 6, 7, 8, 9. |
| `mask_indices_by_tp` | MoE | yes | yes | generated | pending | Scalar int64 PTO port for TP-local expert remapping, preserving `-1` sentinel behavior. |
| `group_count` | MoE | yes | yes | generated | pending | Correctness-first single-block scalar scan; avoids atomics for validation, not a final parallel reduction. |
| `aux_fi` | MoE | yes | yes | generated | pending | Correctness-first single-block scalar scan with f32 scaling by `num_experts / (num_tokens * num_aux_topk)`. |
| `topk_gate` | MoE | yes | yes | generated | pending | Uses PTO-ISA-style `tfillpad_expand` with `PadValue::Min`, `sort32`, `mrgsort`, and `gather` for all TileKernels expert configs. Validation currently emits uint32 indices from PTO sort lanes; the TileKernels torch API widens indices to int64. |
| `topk_sum_and_topk_group_idx` | MoE | yes | yes | generated | pending | Uses grouped PTO sort/pad/gather twice: first to sum the top expert scores inside each group, then to select the top groups from per-group sums. Validation currently emits uint32 group indices from PTO sort lanes; the TileKernels torch API widens indices to int64. |
| `inplace_unique_group_indices` | MoE | yes | yes | generated | pending | Correctness-first scalar row scan over int64 group ids; later duplicate non-negative ids are replaced with `-1`. |
| `transpose` | Transpose | yes | yes | generated | pending | Dynamic 2-D bf16/f32 transpose using PTO `TTrans`; runtime grid tracks `num_tokens={0,4032,8064}` and hidden sizes `{576,2048,2560,3072,4096,6144,7168}`. |
| `batched_transpose` | Transpose | yes | yes | generated | pending | Dynamic batched bf16/f32 transpose for expert counts `{8,32}` over the same token/hidden grid. |
| `fused_weight` | Engram | yes | yes | generated | pending | bf16 `weight_hidden * weight_embed` with f32 output for `hc=4` and hidden sizes `{2048,2560,3072,4096,6144,7168}`. |
| `engram_hash` | Engram | yes | yes | generated | pending | Scalar/token-parallel int hash with int64 multipliers, vocab modulo, and offsets for `max_ngram_size=3`, `num_ngram_layers=2`, and 8 tables per n-gram. |
| `engram_gate_fwd` | Engram | yes | yes | generated | pending | Fixed-hidden forward save path for hidden `{2048,4096,7168}` using f32 RMS/dot row reductions, tile signed-sqrt sigmoid, bf16 output, and saved dot/gate/rstd intermediates. |
| `engram_gate_bwd` | Engram | yes | yes | generated | pending | Correctness-first backward for hidden `{2048,4096,7168}` and 4 persistent blocks; emits bf16 grad_x/grad_k/grad_v plus f32 grad_w_partial. |
| `grad_w_reduce` | Engram | yes | yes | generated | pending | f32 reduction over persistent partial gradients, bf16-to-f32 weight conversion, and in-place f32 gradient accumulation. |
| `expand_to_mhc_fwd` | MHC | yes | yes | generated | pending | bf16 forward expansion over `n0={1,2}`, `n1={1024,4096}`, `mhc_mult={2,4,8}`, hidden `{1280,2560,7168}`. |
| `expand_to_mhc_bwd` | MHC | yes | yes | generated | pending | bf16 backward reduction across the MHC axis with f32 tile accumulation before bf16 output. |
| `sinkhorn_normalize_fwd` | MHC | yes | yes | generated | pending | Tile f32 Sinkhorn forward for `n0={1,2}`, `n1={1,1024,4096}`, `mhc=4`, `repeat=10`, and `eps=1e-6`; scalar `math.exp` was replaced with PTO `texp`. |
| `sinkhorn_normalize_bwd` | MHC | yes | yes | generated | pending | Tile f32 reverse-mode Sinkhorn backward over the same shape grid, storing forward stage tiles and applying row/column normalization adjoints in reverse. |
| `head_compute_mix_fwd` | MHC | yes | yes | generated | pending | f32 sigmoid head-mix forward using PTO `texp`/`trecip` instead of scalar `math.exp`. |
| `head_compute_mix_bwd` | MHC | yes | yes | generated | pending | Correctness-first single-block backward for input gradients and one partial scale/base gradient row. |
| `pre_split_mixes_fwd` | MHC | yes | yes | generated | pending | f32 pre/post/comb split from 24-wide mix rows using 4-wide and 16-wide tile subsets. |
| `pre_split_mixes_bwd` | MHC | yes | yes | generated | pending | Correctness-first single-block backward for input gradients plus partial scale/base gradients. |
| `pre_apply_mix_fwd` | MHC | yes | yes | generated | pending | bf16 MHC-lane weighted hidden reduction with f32 accumulation. |
| `pre_apply_mix_bwd` | MHC | yes | yes | generated | pending | Correctness-first backward updating `x_grad` and producing per-token `mix_grad`. |
| `post_fwd` | MHC | yes | yes | generated | pending | bf16 post-layer x plus comb-residual branch with f32 accumulation. |
| `post_bwd` | MHC | yes | yes | generated | pending | Correctness-first backward for x, residual, post mix, and comb mix gradients. |
| `pre_norm_fn_fwd` | MHC | yes | yes | generated | pending | Baseline RMS-normalized residual/FN projection forward without optional norm-weight merge. |
| `fn_normw_merge_fwd` | MHC | yes | yes | generated | pending | Optional pre-norm FN/norm-weight merge forward. |
| `fn_normw_merge_bwd` | MHC | yes | yes | generated | pending | Optional pre-norm FN/norm-weight merge backward for FN and norm-weight gradients. |

The generated validation `main.cpp`, `golden.py`, and `compare.py` contain host
allocation, kernel launch, output capture, and NumPy comparison logic for
deterministic runtime shapes per compiled PTO-DSL variant. Local
`validation_compile`, SIM correctness, and NPU correctness remain pending
because this macOS host does not provide the Ascend/CANN compiler/runtime.

## Full Checklist

Gate fields in `kernel_status.json`:

- `pto_dsl_source`
- `ptoas`
- `validation_assets`
- `validation_compile`
- `sim_correct`
- `npu_correct`

| Family | Kernels |
| --- | --- |
| MoE | `topk_gate`, `topk_sum_and_topk_group_idx`, `top2_sum_gate`, `normalize_weight`, `group_count`, `aux_fi`, `mask_indices_by_tp`, `inplace_unique_group_indices`, `get_fused_mapping`, `expand_to_fused`, `expand_to_fused_with_sf`, `reduce_fused` |
| Quant | `per_token_cast`, `per_block_cast`, `per_channel_cast`, `per_channel_cast_fused`, `per_channel_cast_and_transpose`, `swiglu_forward_and_per_token_cast`, `swiglu_backward_and_per_token_cast`, `swiglu_forward_and_per_channel_cast_and_transpose`, `cast_back`, `per_token_cast_back`, `per_block_cast_lossless`, E5M6 paths |
| Transpose | `transpose`, `batched_transpose` |
| Engram | `engram_hash`, `fused_weight`, `engram_gate_fwd`, `engram_gate_bwd`, `grad_w_reduce` |
| MHC | `expand_to_mhc`, `sinkhorn_normalize`, `mhc_pre_split_mixes`, `mhc_pre_apply_mix`, `mhc_pre_norm_fn`, `mhc_pre_big_fuse`, `mhc_head_compute_mix`, `mhc_post`, `mhc_multilayer_recompute` |

## Migration Notes

- TileLang CUDA warp concepts such as lane IDs, shared memory reductions,
  atomics, `__match_any_sync`, and PTX wait groups must be redesigned with PTO
  tile/vector operations. They should not be transliterated. Current
  `group_count` and `aux_fi` ports are scalar correctness bridges, not the final
  parallel algorithms.
- `topk_gate` and `topk_sum_and_topk_group_idx` now start from PTO tile sort
  primitives (`tfillpad_expand`, `sort32`, `mrgsort`, `gather`) rather than
  scalar scans. PTO-Gym validation emits uint32 indices for these ports because
  PTO sort exposes uint32 index lanes; TileKernels' public torch API widens
  indices to int64. Remaining TopK-family work is `top2_sum_gate`.
- `get_fused_mapping` still needs a different design: a scalar scan prototype
  produced valid mapping logic, but local `ptoas` rejected the data-dependent
  scalar stores into `pos_to_*` arrays. Do not mark this port complete until the
  placement step is expressed with PTO-supported tile/scatter operations or the
  backend supports that store pattern.
- Engram gate forward/backward now follows the PTO-kernels RMSNorm pattern:
  regular row-tile loads, f32 row reductions for RMS/dot, tile select for the
  signed-sqrt gate, and per-block partial weight gradients. `grad_w_reduce`
  remains the final accumulation stage for weight_hidden/weight_embed grads.
- Quant kernels need an explicit policy for FP4, E5M6, packed scale factors,
  and any format not directly represented by PTO-DSL types.
- MHC fused kernels should be ported as smaller verified stages first, then
  fused after each stage has ptoas and correctness evidence. The current split
  stages with local `ptoas` coverage are `head_compute_mix_{fwd,bwd}`,
  `pre_split_mixes_{fwd,bwd}`, `pre_apply_mix_{fwd,bwd}`, and
  `post_{fwd,bwd}`. `pre_norm_fn_fwd` now covers the baseline projection, and
  `fn_normw_merge_{fwd,bwd}` covers the optional norm-weight path; pre-norm
  backward remains a separate follow-up slice.

## Commands

List registered PTO-DSL cases:

```bash
PTODSL_ROOT=/Users/zhoubot/github/pto-org/pto-dsl \
PYTHONPATH=/Users/zhoubot/github/.llvm-19.1.7/build-mlir-py312/tools/mlir/python_packages/mlir_core:/Users/zhoubot/github/pto-org/PTOAS/build-src312/python:/Users/zhoubot/github/pto-org/pto-dsl \
python3.12 examples/ptodsl/tilekernels/compile_kernels.py --list
```

Generate `.pto` without ptoas:

```bash
PTODSL_ROOT=/Users/zhoubot/github/pto-org/pto-dsl \
PYTHONPATH=/Users/zhoubot/github/.llvm-19.1.7/build-mlir-py312/tools/mlir/python_packages/mlir_core:/Users/zhoubot/github/pto-org/PTOAS/build-src312/python:/Users/zhoubot/github/pto-org/pto-dsl \
python3.12 examples/ptodsl/tilekernels/compile_kernels.py --case moe.normalize_weight --skip-ptoas
```

Run ptoas and materialize PTO-Gym validation assets when the toolchain is
available:

```bash
PTODSL_ROOT=/Users/zhoubot/github/pto-org/pto-dsl \
PTOAS_BIN=/Users/zhoubot/github/pto-org/PTOAS/build-src312/tools/ptoas/ptoas \
PTOAS_FLAGS="--pto-arch a5" \
PYTHONPATH=/Users/zhoubot/github/.llvm-19.1.7/build-mlir-py312/tools/mlir/python_packages/mlir_core:/Users/zhoubot/github/pto-org/PTOAS/build-src312/python:/Users/zhoubot/github/pto-org/pto-dsl \
python3.12 examples/ptodsl/tilekernels/compile_kernels.py --materialize-validation
```

Run PTO-Gym's host validation script against generated TileKernels cases in an
Ascend/CANN environment:

```bash
CASES_ROOT=$PWD/.work/ptodsl-tilekernels/validation \
WORK_SPACE=$PWD/.work/vpto-tilekernels \
ASCEND_HOME_PATH=/path/to/ascend \
PTOAS_BIN=/Users/zhoubot/github/pto-org/PTOAS/build-src312/tools/ptoas/ptoas \
PTOAS_FLAGS="--pto-arch a5 --enable-insert-sync" \
COMPILE_ONLY=1 \
examples/pto/scripts/run_host_vpto_validation.sh
```
