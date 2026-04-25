# `engram.fused_weight`

TileKernels source: `/Users/zhoubot/TileKernels/tile_kernels/engram/`

This PTO-DSL port implements the deterministic fused-weight path:

```python
weight_fused = weight_hidden.astype("float32") * weight_embed.astype("float32")
```

Implemented compile config:

- `hc=4`
- `input_dtype=bf16`
- `output_dtype=f32`

The shape manifest tracks TileKernels hidden sizes
`{2048,2560,3072,4096,6144,7168}`. Full SIM/NPU correctness is pending host
validation in an Ascend/CANN environment.
