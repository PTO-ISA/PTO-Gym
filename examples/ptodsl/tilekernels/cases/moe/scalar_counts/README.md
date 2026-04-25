# MoE Scalar Count And Mask Kernels

TileKernels sources:

- `/Users/zhoubot/TileKernels/tile_kernels/moe/mask_indices_by_tp_kernel.py`
- `/Users/zhoubot/TileKernels/tile_kernels/moe/group_count_kernel.py`
- `/Users/zhoubot/TileKernels/tile_kernels/moe/aux_fi_kernel.py`

Implemented PTO-DSL builders:

- `moe.mask_indices_by_tp`: int64 expert index masking/remapping for a TP rank.
- `moe.group_count`: int64 expert indices to int32 per-expert counts.
- `moe.aux_fi`: int64 expert indices to f32 auxiliary frequency indicators.

`group_count` and `aux_fi` are correctness-first scalar scans that run with one
PTO block and avoid TileLang atomics. They are real validation kernels, but not
the final throughput-oriented algorithms.
