# `moe.normalize_weight`

TileKernels source: `/Users/zhoubot/TileKernels/tile_kernels/moe/normalize_weight_kernel.py`

This seed port implements row-wise top-k weight normalization for float32 input
rows. The output denominator is represented as a `[num_tokens, 1]` tensor view
inside the generated PTO kernel; Python callers may expose it as a flat
`[num_tokens]` tensor.

Implemented compile configs:

- `num_topk=1`
- `num_topk=2`
- `num_topk=6`
- `num_topk=7`
- `num_topk=8`
- `num_topk=9`

The harness can also materialize PTO-Gym validation assets for each config under
`.work/ptodsl-tilekernels/validation/moe/normalize_weight`.
Those generated assets include host runner and NumPy golden/compare logic for a
deterministic `num_tokens=4001` runtime shape. Full SIM/NPU correctness is still
pending an Ascend/CANN validation run.

Correctness target:

```python
denominator = 1e-20 + topk_weights.sum(dim=1)
normalized = topk_weights / denominator[:, None]
```
