<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# VMI Kernel 用例

本目录只保留从目标仓库 `.work/external/a5-kernel-standalone/cce` 的 raw CCE
正确性、等价性和 minimum 测试入口迁移而来的 VMI runtime case。
范围定义见 [CCE_CASE_SCOPE.md](CCE_CASE_SCOPE.md)。

不要从历史 VMI probe、benchmark sweep、debug shape、random stress 或当前目录外的
实验脚本反推支持范围。新增 case 前先确认目标 CCE 测试入口是否提供对应正确性语义。

## 当前目录范围

当前目录保留 36 个 runtime case：

| CCE family | 当前 case 数 | 范围 |
| --- | ---: | --- |
| `quant_minimum` / `tquant` | 4 | 对齐 `MINIMUM_CASES` |
| `block_quant` | 7 | 对齐 `test_equivalence.py` |
| `dynamic_quant` | 9 | 对齐 `test_dq_equivalence.py` |
| `dequant/anti_mx_quant` | 7 | 当前保留 VMI 能表达的 FP8 行；FP4 输入暂缓 |
| `block_mx_quant` | 4 | 当前保留 canonical FP8/OCP 等价性行；FP4、DDR `scale_alg=2` 和额外 `test_cce.py` union 行暂缓 |
| `swiglu_mx_quant` | 4 | 当前保留 FP8/OCP 等价性行；FP4 和 CCE 已标记异常的 CUBLAS `scale_alg=1` 暂缓 |
| `simdvf_per_block_cast` | 1 | 对齐 PTOAS PR #488 中的 16x256 f16 + 4x8 scale -> fp8 per-block cast case |

## 设计上暂缓

下列目标 CCE 行是真实存在的，但在对应 VMI 语义设计清楚前，不应通过临时拼凑的
runtime case 表达：

| Case 类别 | 原因 |
| --- | --- |
| FP4 packed input/output | VMI 尚未定义 logical FP4 lane、packed-byte layout 和 FP4 load/store 语义 |
| `block_mx_quant` FP4 DDR `scale_alg=2` | 依赖 FP4 语义和 DDR scale 规则 |
| `swiglu_mx_quant` FP8 CUBLAS `scale_alg=1` | CCE 源码已标记该路径异常 |
| HIF8 `block_quant` | 只是 compiler/runtime surface probe，不是目标仓库里的 raw CCE 正确性 case |

## 验证策略

每个保留的 runtime case 都应通过 `test/vpto/scripts/run_host_vpto_validation.sh`。
新增 CCE 迁移 case 时，先在 [CCE_CASE_SCOPE.md](CCE_CASE_SCOPE.md) 记录对应的
目标 CCE 来源行。除非新 case 覆盖不同的 VMI 语义或 lowering 约束，否则不要重复添加同构 shape。
