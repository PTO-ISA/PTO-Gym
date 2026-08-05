<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# VMI Kernels 的 CCE Case 范围

本文档从目标仓库 `.work/external/a5-kernel-standalone/cce` 的 raw CCE 测试入口列出
VMI kernel 迁移范围。当前审计快照为本地 clone 的
`main@ee81c3660d6336ecaecd805f02ffb2d69446984e`。不要从当前
`test/vpto/cases/vmi_new/kernels` 已存在目录反推目标范围：历史目录包含额外 probe、
历史 VMI coverage 和尚未对齐 CCE 数据流的 case。

## 统计规则

“必须支持”只统计目标仓库中的正确性、等价性和 minimum 测试入口。
`smoke`、`timing`、`bench`、`debug`、`experiments` 和 bandwidth sweep 只用于补充代表 shape
或性能验证，不自动扩展为必须支持的语义 case。

| CCE family | 正确性来源 | 必须支持数量 | 首批 VMI 目标 | 暂缓或非首批 |
| --- | --- | ---: | --- | --- |
| `quant_minimum` | `quant_minimum/test/test_tquant.py` | 4 | 全部 4 个 | 该 suite 无暂缓项 |
| `block_quant` | `block_quant/test/test_equivalence.py` | 7 | 全部 7 个 | 除非 raw CCE 新增正确性入口，否则 HIF8 只算 VMI/compiler probe |
| `dynamic_quant` | `dynamic_quant/test/test_dq_equivalence.py` | 9 | 全部 9 个 | subset wrapper 不新增语义 case |
| `dequant/anti_mx_quant` | `dequant/anti_mx_quant/test/test_equivalence.py` | 16 | 先支持 FP8 case | FP4 输入因 VMI FP4 surface 未设计而暂缓 |
| `block_mx_quant` | `block_mx_quant/test/test_equivalence.py`; `test_cce.py` 是更宽的 smoke/correctness surface | 14 canonical，30 full union | 先支持 canonical FP8/OCP/rint 行 | FP4、DDR `scale_alg=2` 和额外 `test_cce.py` union 行暂缓 |
| `swiglu_mx_quant` | `swiglu_mx_quant/test/test_equivalence.py` | 14 | 先支持 FP8/OCP/rint f16/bf16 行 | FP4 暂缓；CCE 源码中 `scale_alg=1` CUBLAS 路径异常 |
| `simdvf_per_block_cast` | PTOAS PR #488 | 1 | 支持 16x256 f16 + 4x8 f32 scale -> fp8 cast | 当前只迁移该 PR 提供的确定 shape |
| `tutorial/block_mx_quant` | `tutorial/block_mx_quant/README.md` | 已由 `block_mx_quant` 覆盖 | BF16 FP8 tutorial shape 作为代表覆盖 | tutorial FP4 与主 `block_mx_quant` 共用 FP4 blocker |

## quant_minimum

来源：`.work/external/a5-kernel-standalone/cce/quant_minimum/test/test_tquant.py`。

| 目标 case | VMI 支持状态 |
| --- | --- |
| `mxfp8_32x32_nd` | 必须支持 |
| `mxfp8_32x64_nz` | 必须支持 |
| `int8_sym_64x128_nd` | 必须支持 |
| `int8_asym_64x128_nd` | 必须支持 |

`test_cycle_match.py` 只对同一组 `MINIMUM_CASES` 做 PTO/CCE cycle 对比，不新增语义 case。

## block_quant

来源：`.work/external/a5-kernel-standalone/cce/block_quant/test/test_equivalence.py`。
所有 case 都使用 `row_block_size=1`、`col_block_size=128`、`dst_type=292`。

| 目标 case | VMI 支持状态 |
| --- | --- |
| BF16 `(4,128)` | 必须支持 |
| FP16 `(8,128)` | 必须支持 |
| BF16 `(32,128)` | 必须支持 |
| FP16 `(16,256)` | 必须支持 |
| BF16 `(2,128)` | 必须支持 |
| FP16 `(4,256)` | 必须支持 |
| 带 `min_scale` 的 BF16 `(4,128)` | 必须支持 |

`minimal_test.py` 是 smoke 子集。`test_hardware_scale.py`、`large_shape_correctness.py`
和 bandwidth sweep 主要验证更大 streaming shape；只有当它们暴露新的 VMI memory/layout
约束时，才增加代表性 runtime case。

## simdvf per-block cast to FP8

来源：PTOAS PR #488。

| 目标 case | VMI 支持状态 |
| --- | --- |
| FP16 input `(16,256)`，FP32 scale `(4,8)`，每个 scale 覆盖 4 行 x 32 列，输出 FP8 E4M3 | 必须支持 |

## dynamic_quant

来源：`.work/external/a5-kernel-standalone/cce/dynamic_quant/test/test_dq_equivalence.py`。

| 目标 case | VMI 支持状态 |
| --- | --- |
| per-token，无 smooth，FP16 `(4,32)` | 必须支持 |
| per-token，无 smooth，FP16 `(16,128)` | 必须支持 |
| per-token，smooth，FP16 `(8,64)` | 必须支持 |
| per-token，smooth，FP16 `(16,128)` | 必须支持 |
| per-channel，FP16 `(128,128)` | 必须支持 |
| per-channel，FP16 `(256,256)` | 必须支持 |
| per-token，无 smooth，BF16 `(4,32)` | 必须支持 |
| per-token，smooth，BF16 `(8,64)` | 必须支持 |
| per-channel，BF16 `(128,128)` | 必须支持 |

`test_pertoken_only.py`、`test_perchannel_128.py` 和 `test_perchannel_all.py`
只是该表的子集或重新分组。

## dequant / anti_mx_quant

来源：`.work/external/a5-kernel-standalone/cce/dequant/anti_mx_quant/test/test_equivalence.py`。

| 目标 case | VMI 支持状态 |
| --- | --- |
| FP8 E4M3 -> BF16 `(4,128)` | 首批必须支持 |
| FP8 E4M3 -> FP32 `(4,128)` | 首批必须支持 |
| FP8 E4M3 -> FP16 `(4,128)` | 首批必须支持 |
| FP8 E4M3 -> BF16 `(16,512)` | 首批必须支持 |
| FP8 E4M3 -> BF16 `(64,2048)` | 首批必须支持 |
| FP8 E4M3 -> BF16 `(1024,2048)` | 代表 large/perf；若 medium 已覆盖相同 lowering，首批 runtime shape 不必加入 |
| FP8 E5M2 -> BF16 `(4,128)` | 首批必须支持 |
| FP8 E5M2 -> BF16 `(16,512)` | 首批必须支持 |
| FP4 E2M1 -> BF16 `(4,64)` | 暂缓 |
| FP4 E2M1 -> BF16 `(16,256)` | 暂缓 |
| FP4 E2M1 -> BF16 `(4096,512)` | 暂缓 |
| FP4 E2M1 -> BF16 `(65536,2048)` | 暂缓 |
| FP4 E1M2 -> BF16 `(4,64)` | 暂缓 |
| FP4 E1M2 -> BF16 `(16,256)` | 暂缓 |
| FP4 E1M2 -> BF16 `(4096,512)` | 暂缓 |
| FP4 E1M2 -> BF16 `(65536,2048)` | 暂缓 |

这些 FP4 行是真实目标仓库 case，但当前 VMI 尚未定义 logical FP4 packed input lane
或 packed-byte load/store 语义。不要用临时 byte trick 模拟。

## block_mx_quant

canonical 来源：`.work/external/a5-kernel-standalone/cce/block_mx_quant/test/test_equivalence.py`。
这是 `HW_RESULTS.md` 中报告的默认 14-case 正确性 suite。

| 目标 case | VMI 支持状态 |
| --- | --- |
| FP8 E4M3 BF16 `(4,128)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP8 E4M3 FP16 `(64,256)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP8 E5M2 BF16 `(4,128)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP8 E5M2 FP16 `(8,256)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP4 E2M1 BF16 `(4,128)`, `scale_alg=0`, `rint` | 暂缓 |
| FP4 E2M1 BF16 `(256,512)`, `scale_alg=0`, `round` | 暂缓 |
| FP4 E2M1 FP16 `(4,128)`, `scale_alg=0`, `floor` | 暂缓 |
| FP4 E2M1 BF16 `(1,2,2)`, `scale_alg=2`, `floor`, `dst_type_max=0` | 暂缓 |
| FP4 E2M1 BF16 `(4,128)`, `scale_alg=2`, `floor`, `dst_type_max=6` | 暂缓 |
| FP4 E2M1 BF16 `(4,128)`, `scale_alg=2`, `rint`, `dst_type_max=7` | 暂缓 |
| FP4 E1M2 BF16 `(4,128)`, `scale_alg=0`, `rint` | 暂缓 |
| FP4 E1M2 FP16 `(8,256)`, `scale_alg=0`, `round` | 暂缓 |
| FP4 E1M2 BF16 `(4,128)`, `scale_alg=0`, `floor` | 暂缓 |
| FP4 tail BF16 `(100,300)`, E2M1, `scale_alg=0`, `rint` | 暂缓 |

`test_cce.py` 额外枚举完整 small-shape type/rounding union：

| Surface family | 额外覆盖 |
| --- | --- |
| FP8 OCP | FP16/BF16 x E4M3/E5M2, shape `(4,128)`, `rint` |
| FP4 E2M1 OCP | FP16/BF16 x `rint/round/floor`, shape `(4,128)` |
| FP4 E2M1 DDR | FP16/BF16 x `rint/round/floor`, shape `(4,128)`, `scale_alg=2` |
| FP4 E1M2 OCP | FP16/BF16 x `rint/round/floor`, shape `(4,128)` |

VMI 实现以 canonical 14-case suite 作为迁移 checklist；`test_cce.py` union
在 FP4 设计完成后作为 surface 完整性 checklist。

## swiglu_mx_quant

来源：`.work/external/a5-kernel-standalone/cce/swiglu_mx_quant/test/test_equivalence.py`。

| 目标 case | VMI 支持状态 |
| --- | --- |
| FP8 E4M3 BF16 `(4,8)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP8 E4M3 FP16 `(64,512)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP8 E5M2 BF16 `(4,8)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP8 E5M2 FP16 `(128,256)`, `scale_alg=0`, `rint` | 首批必须支持 |
| FP8 E4M3 BF16 `(64,512)`, `scale_alg=1`, `rint` | 暂缓；CCE 标记 CUBLAS 路径异常 |
| FP8 E5M2 FP16 `(64,512)`, `scale_alg=1`, `rint` | 暂缓；CCE 标记 CUBLAS 路径异常 |
| FP4 E2M1 BF16 `(4,8)`, `scale_alg=0`, `rint` | 暂缓 |
| FP4 E2M1 FP16 `(4,8)`, `scale_alg=0`, `rint` | 暂缓 |
| FP4 E2M1 BF16 `(64,512)`, `scale_alg=0`, `round` | 暂缓 |
| FP4 E2M1 BF16 `(4,8)`, `scale_alg=0`, `floor` | 暂缓 |
| FP4 E2M1 BF16 `(128,256)`, `scale_alg=0`, `rint` | 暂缓 |
| FP4 E1M2 BF16 `(4,8)`, `scale_alg=0`, `rint` | 暂缓 |
| FP4 E1M2 FP16 `(64,512)`, `scale_alg=0`, `round` | 暂缓 |
| FP4 E1M2 BF16 `(4,8)`, `scale_alg=0`, `floor` | 暂缓 |

`test_smoke.py` 在 shape `(4,8)`、`(64,512)`、`(128,256)`、dtype BF16/FP16、
FP4/FP8 输出模式和 FP8 `scale_alg=1` 上跑 48 个执行面。它不是正确性 oracle；
只有在等价性 case 覆盖后才使用。

`test_constant_input.py` 用于诊断 `(4,8)` 和 `(64,512)` 上 BF16 E4M3 OCP
constant input。它可以支撑 tiny deterministic VMI case，但除非 equivalence suite
新增对应项，否则不应为每个 dtype/output type 建一套并行矩阵。

## tutorial / block_mx_quant

来源：`.work/external/a5-kernel-standalone/cce/tutorial/block_mx_quant/README.md`。

tutorial kernel 是教学用途，和主 `block_mx_quant` 共享算法：BF16 输入、`scale_alg=0`、
FP8/FP4 输出、32x32 block scale 和 scale2 interleaving。README 说明 smoke 有 9 个
BF16 output-type case，cross-check 有 7 个 byte-exact case，但当前快照中没有详细测试文件。
因此 tutorial 覆盖只作为主 `block_mx_quant` 表的代表 shape，不作为独立 family。

## 当前 VMI 目录裁剪结果

该目录已裁剪为 target-scoped runtime case。删除的 case 只有在目标仓库新增匹配的正确性入口，
或迁移到独立的非目标 probe suite 后，才应重新引入。

当前 `test/vpto/cases/vmi_new/kernels` 已缩减为 36 个 case 目录。上面的目标 CCE canonical
正确性范围在 `block_mx_quant` 采用 14-case canonical suite 时有 65 行；如果把
`block_mx_quant/test_cce.py` 作为完整 small-shape surface union 计入，则有 81 行。
这些数量不能直接和当前支持集比较，因为目标列表仍包含当前 VMI 有意暂缓的 FP4 行。

| Area | 当前 VMI 目录数 | 目标 canonical 正确性 | 差异 |
| --- | ---: | ---: | --- |
| `quant_minimum` / `tquant` | 4 | 4 | 对齐 `MINIMUM_CASES` |
| `block_quant` | 7 | 7 | 对齐 `test_equivalence.py` |
| `dynamic_quant` | 9 | 9 | 对齐 `test_dq_equivalence.py` |
| `anti_mx_quant` | 7 | 16 | 保留当前 FP8 目标行；暂缓的 FP4 行不表达 |
| `block_mx_quant` | 4 | 14 canonical / 30 full union | 保留 canonical FP8 目标行；暂缓的 FP4/DDR 和额外 `test_cce.py` union 行不表达 |
| `swiglu_mx_quant` | 4 | 14 | 保留当前 FP8/OCP 目标行；暂缓的 FP4 和异常 CUBLAS 行不表达 |
| `simdvf_per_block_cast` | 1 | 1 | 对齐 PR #488 |
| historical `anti_quant` | 0 | 0 | 已从 target-scoped 目录移除 |
| historical `swiglu_quant` | 0 | 0 | 已从 target-scoped 目录移除 |
| other probe | 0 | 0 | 已从 target-scoped 目录移除 |

## 当前支持目录清单

当前 target-scoped runtime 目录精确包含以下 36 个 VMI case：

| CCE family | VMI case 目录 |
| --- | --- |
| `quant_minimum` / `tquant` | `tquant-mxfp8-32x32-nd`, `tquant-mxfp8-32x64-nz`, `tquant-int8-sym-8x128`, `tquant-int8-asym-8x128` |
| `block_quant` | `block-quant-bf16-fp8-2x128`, `block-quant-bf16-fp8-4x128`, `block-quant-bf16-fp8-4x128-min-scale`, `block-quant-bf16-fp8-32x128`, `block-quant-f16-fp8-4x256`, `block-quant-f16-fp8-8x128`, `block-quant-f16-fp8-16x256` |
| `dynamic_quant` | `dynamic-quant-pertoken-f16-4x32`, `dynamic-quant-pertoken-f16-16x128`, `dynamic-quant-pertoken-smooth-f16-8x64`, `dynamic-quant-pertoken-smooth-f16-16x128`, `dynamic-quant-perchannel-f16-8x128`, `dynamic-quant-perchannel-f16-8x256`, `dynamic-quant-pertoken-bf16-4x32`, `dynamic-quant-pertoken-smooth-bf16-8x64`, `dynamic-quant-perchannel-bf16-8x128` |
| `dequant/anti_mx_quant` | `anti-mx-f8-bf16-scaled-4x128`, `anti-mx-f8-f32-scaled-4x128`, `anti-mx-f8-f16-scaled-4x128`, `anti-mx-f8-bf16-scaled-4x512`, `anti-mx-f8-bf16-scaled-2x512`, `anti-mx-f8e5m2-bf16-scaled-4x128`, `anti-mx-f8e5m2-bf16-scaled-4x512` |
| `block_mx_quant` | `block-mx-quant-bf16-e4m3-4x128`, `block-mx-quant-f16-e4m3-2x256`, `block-mx-quant-bf16-e5m2-4x128`, `block-mx-quant-f16-e5m2-8x256` |
| `swiglu_mx_quant` | `swiglu-mx-quant-bf16-e4m3-4x8`, `swiglu-mx-quant-f16-e4m3-2x512`, `swiglu-mx-quant-bf16-e5m2-4x8`, `swiglu-mx-quant-f16-e5m2-2x256` |
| `simdvf_per_block_cast` | `simdvf-per-block-cast-to-fp8` |

| 已移除的 VMI 区域 | 范围说明 |
| --- | --- |
| 额外 `anti-mx` FP8 E5M2 -> FP16/FP32 large-shape case | 对称 decode 覆盖；未列入目标 `anti_mx_quant/test_equivalence.py` |
| 额外 `dynamic_quant` BF16 larger smooth/no-smooth case | 实现扩展覆盖；不在 9-case 目标 equivalence 列表中 |
| 额外 `block_mx_quant` random/shared-scale case | 对独立 golden 覆盖有用；不是直接目标测试名 |
| 额外 `block_mx_quant` FP16 `(4,128)` E4M3/E5M2 行 | 只存在于更宽的 `test_cce.py` union；不是 canonical equivalence checklist 的一部分 |
| 额外 `swiglu_mx_quant` constant BF16 4x8 proxy | 诊断输入模式；除非迁移为精确 equivalence 行，否则不保留在 target-scoped runtime case 中 |
| HIF8 `block_quant` probe | compiler/runtime surface probe；不是该目标仓库中的 raw CCE 正确性 case |
