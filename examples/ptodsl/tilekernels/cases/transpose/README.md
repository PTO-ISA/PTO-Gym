# `transpose`

TileKernels source: `/Users/zhoubot/TileKernels/tile_kernels/transpose/`

Implemented PTO-DSL builders:

- `transpose.transpose`: dynamic `[num_tokens, hidden] -> [hidden, num_tokens]`
  using PTO `TTrans`.
- `transpose.batched_transpose`: dynamic
  `[num_experts, num_tokens, hidden] -> [num_experts, hidden, num_tokens]`
  using PTO `TTrans`.

The registered compile configs cover `bf16` and `f32`. The shape manifest keeps
the deterministic TileKernels correctness grid:

- `num_tokens={0,4032,8064}`
- `hidden={576,2048,2560,3072,4096,6144,7168}`
- batched `num_experts={8,32}`

`fp8_e4m3` remains compile-only/deferred until host byte-pattern comparison is
implemented.
