# `mhc.expand_to_mhc`

TileKernels source: `/Users/zhoubot/TileKernels/tile_kernels/mhc/`

Implemented PTO-DSL builders:

- `mhc.expand_to_mhc_fwd`: bf16 forward expansion,
  `out[n0, n1, mhc, hidden] = x[n0, n1, hidden]`.
- `mhc.expand_to_mhc_bwd`: bf16 backward reduction,
  `x_grad[n0, n1, hidden] = sum(out_grad[n0, n1, mhc, hidden], axis=mhc)`.

The runtime PTO kernel receives `tokens=n0*n1` and `hidden`; the manifest keeps
the original TileKernels shape axes:

- `n0={1,2}`
- `n1={1024,4096}`
- `mhc_mult={2,4,8}`
- `hidden={1280,2560,7168}`

The backward port accumulates in f32 tiles and converts the result back to bf16.
