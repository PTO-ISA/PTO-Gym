#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / ".work" / "ptodsl-tilekernels"
DEFAULT_VALIDATION_DIR = DEFAULT_OUT_DIR / "validation"
DEFAULT_PTOAS_FLAGS = "--pto-arch a5"


def _configure_pythonpath() -> None:
    pto_dsl_root = os.environ.get("PTODSL_ROOT")
    candidates: list[Path] = []
    if pto_dsl_root:
        candidates.append(Path(pto_dsl_root))
    candidates.append(REPO_ROOT.parent / "pto-dsl")

    for candidate in candidates:
        if (candidate / "ptodsl").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))


@dataclass(frozen=True)
class BuildResult:
    case: str
    config: dict[str, Any]
    pto_path: str
    cpp_path: str | None
    ptoas_status: str
    ptoas_command: list[str] | None
    note: str | None = None


def _load_registry():
    _configure_pythonpath()
    from tilekernels_ptodsl.registry import iter_cases

    return list(iter_cases())


def _resolve_ptoas(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_bin = os.environ.get("PTOAS_BIN")
    if env_bin:
        return env_bin
    found = shutil.which("ptoas")
    if found:
        return found
    raise FileNotFoundError("ptoas not found; set PTOAS_BIN or pass --ptoas")


def _case_out_dir(base: Path, case_name: str, config: dict[str, Any]) -> Path:
    config_id = "_".join(f"{key}-{value}" for key, value in sorted(config.items()))
    if not config_id:
        config_id = "default"
    return base / case_name.replace(".", "/") / config_id


def _config_id(config: dict[str, Any]) -> str:
    config_id = "_".join(f"{key}-{value}" for key, value in sorted(config.items()))
    return config_id or "default"


def _build_ir(case, config: dict[str, Any]):
    module_name, fn_name = case.builder.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "mlir":
            raise RuntimeError(
                "PTO-DSL import failed because MLIR Python bindings are unavailable. "
                "Build/install the PTOAS/MLIR Python packages, set PYTHONPATH accordingly, "
                "and set PTODSL_ROOT to the PTO-DSL checkout."
            ) from exc
        raise
    fn = getattr(module, fn_name)
    return fn(**config)


def _run_ptoas(ptoas: str, pto_path: Path, cpp_path: Path, extra_flags: list[str]) -> list[str]:
    cmd = [ptoas]
    if "--enable-insert-sync" not in extra_flags:
        cmd.append("--enable-insert-sync")
    cmd.extend(extra_flags)
    cmd.extend([str(pto_path), "-o", str(cpp_path)])
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if not cpp_path.exists() or cpp_path.stat().st_size == 0:
        raise RuntimeError(f"ptoas did not produce a nonempty C++ file: {cpp_path}")
    return cmd


def _ptoas_flags(cli_flags: list[str]) -> list[str]:
    env_flags = shlex.split(os.environ.get("PTOAS_FLAGS", DEFAULT_PTOAS_FLAGS))
    return env_flags + cli_flags


def _kernel_symbol(ir_text: str) -> str:
    match = re.search(r"func\.func\s+@([A-Za-z_][A-Za-z0-9_]*)", ir_text)
    if not match:
        raise RuntimeError("generated PTO IR does not contain a func.func symbol")
    return match.group(1)


def _case_validation_dir(base: Path, case_name: str, spec: dict[str, Any]) -> Path:
    return base / case_name.replace(".", "/") / str(spec["id"])


def _ctype_for_arg(arg, config: dict[str, Any]) -> str:
    if arg.ctype == "__bf16" and config.get("dtype") == "f32":
        return "float"
    return arg.ctype


def _ptr_decl(arg, config: dict[str, Any]) -> str:
    ctype = _ctype_for_arg(arg, config)
    if arg.pointer:
        return f"{ctype} *{arg.name}"
    return f"{ctype} {arg.name}"


def _gm_decl(arg, config: dict[str, Any]) -> str:
    ctype = _ctype_for_arg(arg, config)
    if arg.pointer:
        return f"__gm__ {ctype} *{arg.name}"
    return f"{ctype} {arg.name}"


def _kernel_call_arg(arg, config: dict[str, Any]) -> str:
    ctype = _ctype_for_arg(arg, config)
    if arg.pointer:
        return f"(__gm__ {ctype} *){arg.name}"
    return arg.name


def _host_call_arg(arg, config: dict[str, Any]) -> str:
    if arg.pointer:
        return f"reinterpret_cast<{_ctype_for_arg(arg, config)} *>({arg.name}_device)"
    return f"{arg.name}_host"


def _dtype_name_for_ctype(ctype: str) -> str:
    if ctype == "float":
        return "f32"
    if ctype == "__bf16":
        return "bf16"
    if ctype == "int32_t":
        return "i32"
    if ctype == "int64_t":
        return "i64"
    if ctype == "uint32_t":
        return "u32"
    raise ValueError(f"unsupported generated validation ctype: {ctype}")


def _dtype_size(dtype: str) -> int:
    if dtype == "f32":
        return 4
    if dtype == "bf16":
        return 2
    if dtype == "i32":
        return 4
    if dtype == "i64":
        return 8
    if dtype == "u32":
        return 4
    raise ValueError(f"unsupported generated validation dtype: {dtype}")


def _full_test_enabled() -> bool:
    return os.environ.get("TK_FULL_TEST") in {"1", "true", "True"}


def _tokens(values: tuple[int, ...], *, alignment: int = 1) -> list[int]:
    aligned = [((value + alignment - 1) // alignment) * alignment for value in values]
    return ([0] if _full_test_enabled() else []) + aligned


def _spec_id(parts: dict[str, Any]) -> str:
    return "_".join(f"{key}-{value}" for key, value in sorted(parts.items())) or "default"


def _topk_sort_width(num_experts: int) -> int:
    width = 32
    while width < num_experts:
        width *= 4
    return width


def _validation_specs(case, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic runtime validation shapes for a compiled variant."""
    name = case.name
    if name == "moe.normalize_weight":
        num_topk = int(config["num_topk"])
        specs = []
        for num_tokens in _tokens((4001,)):
            specs.append({
                "id": _spec_id({"num_topk": num_topk, "tokens": num_tokens}),
                "case": name,
                "config": config,
                "seed": 101 + num_topk + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "topk_weights": {"dtype": "f32", "elements": num_tokens * num_topk, "role": "input"},
                    "denominator": {"dtype": "f32", "elements": num_tokens, "role": "output", "eps": 1e-4},
                    "normalized_weights": {"dtype": "f32", "elements": num_tokens * num_topk, "role": "output", "eps": 1e-4},
                },
                "shape": {"num_tokens": num_tokens, "num_topk": num_topk},
            })
        return specs

    if name == "moe.mask_indices_by_tp":
        num_topk = int(config["num_topk"])
        num_experts = int(config.get("num_experts", 9))
        num_ep_ranks = int(config.get("num_ep_ranks", 8))
        num_tp_ranks = int(config.get("num_tp_ranks", 2))
        per_gpu = num_experts
        per_dp = num_tp_ranks * per_gpu
        specs = []
        tp_ranks = sorted({0, min(1, num_tp_ranks - 1), num_tp_ranks - 1})
        for num_tokens in _tokens((4001,)):
            for tp_rank in tp_ranks:
                specs.append({
                    "id": _spec_id({
                        "num_ep_ranks": num_ep_ranks,
                        "num_experts": num_experts,
                        "num_topk": num_topk,
                        "num_tp_ranks": num_tp_ranks,
                        "tokens": num_tokens,
                        "tp_rank": tp_rank,
                    }),
                    "case": name,
                    "config": config,
                    "seed": 121 + num_topk + num_experts + num_tp_ranks + tp_rank + num_tokens,
                    "scalars": {
                        "per_gpu": per_gpu,
                        "per_dp": per_dp,
                        "num_tp_ranks": num_tp_ranks,
                        "tp_rank": tp_rank,
                        "num_tokens": num_tokens,
                    },
                    "buffers": {
                        "indices": {"dtype": "i64", "elements": num_tokens * num_topk, "role": "input"},
                        "masked_indices": {"dtype": "i64", "elements": num_tokens * num_topk, "role": "output", "eps": 0.0},
                    },
                    "shape": {
                        "num_tokens": num_tokens,
                        "num_topk": num_topk,
                        "num_experts": num_experts,
                        "num_ep_ranks": num_ep_ranks,
                        "n": num_experts * num_ep_ranks,
                    },
                })
        return specs

    if name == "moe.group_count":
        num_topk = int(config["num_topk"])
        num_groups = int(config["num_groups"])
        specs = []
        for num_tokens in _tokens((4001,)):
            specs.append({
                "id": _spec_id({"num_groups": num_groups, "num_topk": num_topk, "tokens": num_tokens}),
                "case": name,
                "config": config,
                "seed": 151 + num_topk + num_groups + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "group_idx": {"dtype": "i64", "elements": num_tokens * num_topk, "role": "input"},
                    "out": {"dtype": "i32", "elements": num_groups, "role": "output", "eps": 0.0},
                },
                "shape": {
                    "num_tokens": num_tokens,
                    "num_topk": num_topk,
                    "num_groups": num_groups,
                },
            })
        return specs

    if name == "moe.aux_fi":
        num_topk = int(config["num_topk"])
        num_experts = int(config["num_experts"])
        specs = []
        for num_tokens in _tokens((4001,)):
            for num_aux_topk in sorted({1, num_topk}):
                specs.append({
                    "id": _spec_id({
                        "num_aux_topk": num_aux_topk,
                        "num_experts": num_experts,
                        "num_topk": num_topk,
                        "tokens": num_tokens,
                    }),
                    "case": name,
                    "config": config,
                    "seed": 181 + num_topk + num_experts + num_aux_topk + num_tokens,
                    "scalars": {"num_aux_topk": num_aux_topk, "num_tokens": num_tokens},
                    "buffers": {
                        "topk_idx": {"dtype": "i64", "elements": num_tokens * num_topk, "role": "input"},
                        "out": {"dtype": "f32", "elements": num_experts, "role": "output", "eps": 1e-6},
                    },
                    "shape": {
                        "num_tokens": num_tokens,
                        "num_topk": num_topk,
                        "num_experts": num_experts,
                        "num_aux_topk": num_aux_topk,
                    },
                })
        return specs

    if name == "moe.topk_gate":
        num_experts = int(config["num_experts"])
        num_topk = int(config["num_topk"])
        sort_ncols = _topk_sort_width(num_experts)
        specs = []
        for num_tokens in _tokens((4001, 8001)):
            specs.append({
                "id": _spec_id({"num_experts": num_experts, "num_topk": num_topk, "tokens": num_tokens}),
                "case": name,
                "config": config,
                "seed": 501 + num_experts + num_topk + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "scores": {"dtype": "f32", "elements": num_tokens * num_experts, "role": "input"},
                    "inidx": {"dtype": "u32", "elements": sort_ncols, "role": "input"},
                    "topk_idx": {"dtype": "u32", "elements": num_tokens * num_topk, "role": "output", "eps": 0.0},
                },
                "shape": {
                    "num_tokens": num_tokens,
                    "num_experts": num_experts,
                    "num_topk": num_topk,
                    "sort_ncols": sort_ncols,
                },
            })
        return specs

    if name == "moe.topk_sum_and_topk_group_idx":
        num_experts = int(config["num_experts"])
        num_groups = int(config["num_groups"])
        num_group_sum_topk = int(config["num_group_sum_topk"])
        num_topk_groups = int(config["num_topk_groups"])
        group_sort_ncols = _topk_sort_width(num_experts // num_groups)
        group_score_sort_ncols = _topk_sort_width(num_groups)
        max_sort_ncols = max(group_sort_ncols, group_score_sort_ncols)
        specs = []
        for num_tokens in _tokens((4001, 8001)):
            specs.append({
                "id": _spec_id({
                    "num_experts": num_experts,
                    "num_groups": num_groups,
                    "num_group_sum_topk": num_group_sum_topk,
                    "num_topk_groups": num_topk_groups,
                    "tokens": num_tokens,
                }),
                "case": name,
                "config": config,
                "seed": 551 + num_experts + num_groups + num_group_sum_topk + num_topk_groups + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "scores": {"dtype": "f32", "elements": num_tokens * num_experts, "role": "input"},
                    "inidx": {"dtype": "u32", "elements": max_sort_ncols, "role": "input"},
                    "group_idx": {"dtype": "u32", "elements": num_tokens * num_topk_groups, "role": "output", "eps": 0.0},
                },
                "shape": {
                    "num_tokens": num_tokens,
                    "num_experts": num_experts,
                    "num_groups": num_groups,
                    "experts_per_group": num_experts // num_groups,
                    "max_sort_ncols": max_sort_ncols,
                    "num_group_sum_topk": num_group_sum_topk,
                    "num_topk_groups": num_topk_groups,
                },
            })
        return specs

    if name == "moe.inplace_unique_group_indices":
        num_topk = int(config["num_topk"])
        num_groups = int(config["num_groups"])
        specs = []
        for num_tokens in (0, 4001):
            specs.append({
                "id": _spec_id({"num_groups": num_groups, "num_topk": num_topk, "tokens": num_tokens}),
                "case": name,
                "config": config,
                "seed": 571 + num_groups + num_topk + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "group_indices": {"dtype": "i64", "elements": num_tokens * num_topk, "role": "inout", "eps": 0.0},
                },
                "shape": {
                    "num_tokens": num_tokens,
                    "num_topk": num_topk,
                    "num_groups": num_groups,
                },
            })
        return specs

    if name == "transpose.transpose":
        dtype = str(config["dtype"])
        specs = []
        for rows in _tokens((4001, 8001), alignment=64):
            for cols in (576, 2048, 2560, 3072, 4096, 6144, 7168):
                specs.append({
                    "id": _spec_id({"dtype": dtype, "hidden": cols, "tokens": rows}),
                    "case": name,
                    "config": config,
                    "seed": (201 if dtype == "bf16" else 202) + rows + cols,
                    "scalars": {"rows": rows, "cols": cols, "stride": cols},
                    "buffers": {
                        "x": {"dtype": dtype, "elements": rows * cols, "role": "input"},
                        "out": {"dtype": dtype, "elements": rows * cols, "role": "output", "eps": 0.0},
                    },
                    "shape": {"rows": rows, "cols": cols},
                })
        return specs

    if name == "transpose.batched_transpose":
        dtype = str(config["dtype"])
        specs = []
        for batches in (8, 32):
            for rows in _tokens((4001, 8001), alignment=64):
                for cols in (576, 2048, 2560, 3072, 4096, 6144, 7168):
                    specs.append({
                        "id": _spec_id({"batches": batches, "dtype": dtype, "hidden": cols, "tokens": rows}),
                        "case": name,
                        "config": config,
                        "seed": (211 if dtype == "bf16" else 212) + batches + rows + cols,
                        "scalars": {"batches": batches, "rows": rows, "cols": cols, "stride": cols},
                        "buffers": {
                            "x": {"dtype": dtype, "elements": batches * rows * cols, "role": "input"},
                            "out": {"dtype": dtype, "elements": batches * rows * cols, "role": "output", "eps": 0.0},
                        },
                        "shape": {"batches": batches, "rows": rows, "cols": cols},
                    })
        return specs

    if name == "engram.fused_weight":
        hc = 4
        specs = []
        for hidden in (2048, 2560, 3072, 4096, 6144, 7168):
            specs.append({
                "id": _spec_id({"hc": hc, "hidden": hidden}),
                "case": name,
                "config": config,
                "seed": 301 + hidden,
                "scalars": {"hc": hc, "hidden": hidden},
                "buffers": {
                    "weight_hidden": {"dtype": "bf16", "elements": hc * hidden, "role": "input"},
                    "weight_embed": {"dtype": "bf16", "elements": hc * hidden, "role": "input"},
                    "weight_fused": {"dtype": "f32", "elements": hc * hidden, "role": "output", "eps": 1e-4},
                },
                "shape": {"hc": hc, "hidden": hidden},
            })
        return specs

    if name == "engram.engram_hash":
        max_ngram_size = int(config["max_ngram_size"])
        num_ngram_layers = int(config["num_ngram_layers"])
        num_embed_table_per_ngram = int(config["num_embed_table_per_ngram"])
        num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram
        specs = []
        for num_tokens in (0, 4001):
            specs.append({
                "id": _spec_id({
                    "layers": num_ngram_layers,
                    "ngram": max_ngram_size,
                    "tables": num_embed_table_per_ngram,
                    "tokens": num_tokens,
                }),
                "case": name,
                "config": config,
                "seed": 321 + max_ngram_size + num_ngram_layers + num_embed_table_per_ngram + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "ngram_token_ids": {"dtype": "i32", "elements": num_tokens * max_ngram_size, "role": "input"},
                    "multipliers": {"dtype": "i64", "elements": num_ngram_layers * max_ngram_size, "role": "input"},
                    "vocab_sizes": {"dtype": "i32", "elements": num_ngram_layers * (max_ngram_size - 1) * num_embed_table_per_ngram, "role": "input"},
                    "offsets": {"dtype": "i32", "elements": num_ngram_layers * num_out_cols, "role": "input"},
                    "output": {"dtype": "i32", "elements": num_ngram_layers * num_tokens * num_out_cols, "role": "output", "eps": 0.0},
                },
                "shape": {
                    "num_tokens": num_tokens,
                    "max_ngram_size": max_ngram_size,
                    "num_ngram_layers": num_ngram_layers,
                    "num_embed_table_per_ngram": num_embed_table_per_ngram,
                    "num_out_cols": num_out_cols,
                },
            })
        return specs

    if name == "engram.grad_w_reduce":
        num_persistent_blocks = int(config["num_persistent_blocks"])
        hc_mult = int(config["hc_mult"])
        specs = []
        for hidden in (2048, 4096, 7168):
            specs.append({
                "id": _spec_id({
                    "hc": hc_mult,
                    "hidden": hidden,
                    "persistent": num_persistent_blocks,
                }),
                "case": name,
                "config": config,
                "seed": 341 + hidden + num_persistent_blocks + hc_mult,
                "scalars": {"hidden": hidden},
                "buffers": {
                    "grad_w_partial": {"dtype": "f32", "elements": num_persistent_blocks * hc_mult * hidden, "role": "input"},
                    "weight_hidden": {"dtype": "bf16", "elements": hc_mult * hidden, "role": "input"},
                    "weight_embed": {"dtype": "bf16", "elements": hc_mult * hidden, "role": "input"},
                    "grad_weight_hidden": {"dtype": "f32", "elements": hc_mult * hidden, "role": "inout", "eps": 1e-3},
                    "grad_weight_embed": {"dtype": "f32", "elements": hc_mult * hidden, "role": "inout", "eps": 1e-3},
                },
                "shape": {
                    "num_persistent_blocks": num_persistent_blocks,
                    "hc_mult": hc_mult,
                    "hidden": hidden,
                },
            })
        return specs

    if name == "engram.engram_gate_fwd":
        hidden = int(config["hidden_size"])
        hc_mult = int(config["hc_mult"])
        eps = float(config.get("eps", 1e-20))
        clamp_value = float(config.get("clamp_value", 1e-6))
        specs = []
        for num_tokens in (0, 17):
            specs.append({
                "id": _spec_id({"hc": hc_mult, "hidden": hidden, "tokens": num_tokens}),
                "case": name,
                "config": config,
                "seed": 361 + hidden + hc_mult + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "hidden_states": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "input"},
                    "k": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "input"},
                    "v": {"dtype": "bf16", "elements": num_tokens * hidden, "role": "input"},
                    "weight_fused": {"dtype": "f32", "elements": hc_mult * hidden, "role": "input"},
                    "output": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "output", "eps": 1e-2},
                    "dot_out": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "output", "eps": 1e-2},
                    "gate_score": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "output", "eps": 1e-4},
                    "rstd_x": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "output", "eps": 1e-4},
                    "rstd_k": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "output", "eps": 1e-4},
                },
                "shape": {
                    "num_tokens": num_tokens,
                    "hc_mult": hc_mult,
                    "hidden": hidden,
                    "eps": eps,
                    "clamp_value": clamp_value,
                },
            })
        return specs

    if name == "engram.engram_gate_bwd":
        hidden = int(config["hidden_size"])
        hc_mult = int(config["hc_mult"])
        num_persistent_blocks = int(config["num_persistent_blocks"])
        clamp_value = float(config.get("clamp_value", 1e-6))
        specs = []
        for num_tokens in (0, 17):
            specs.append({
                "id": _spec_id({
                    "hc": hc_mult,
                    "hidden": hidden,
                    "persistent": num_persistent_blocks,
                    "tokens": num_tokens,
                }),
                "case": name,
                "config": config,
                "seed": 381 + hidden + hc_mult + num_persistent_blocks + num_tokens,
                "scalars": {"num_tokens": num_tokens},
                "buffers": {
                    "grad_out": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "input"},
                    "hidden_states": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "input"},
                    "k": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "input"},
                    "v": {"dtype": "bf16", "elements": num_tokens * hidden, "role": "input"},
                    "weight_fused": {"dtype": "f32", "elements": hc_mult * hidden, "role": "input"},
                    "dot_in": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "input"},
                    "gate_in": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "input"},
                    "rstd_x_in": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "input"},
                    "rstd_k_in": {"dtype": "f32", "elements": num_tokens * hc_mult, "role": "input"},
                    "grad_x": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "output", "eps": 2e-2},
                    "grad_k": {"dtype": "bf16", "elements": num_tokens * hc_mult * hidden, "role": "output", "eps": 2e-2},
                    "grad_v": {"dtype": "bf16", "elements": num_tokens * hidden, "role": "output", "eps": 2e-2},
                    "grad_w_partial": {"dtype": "f32", "elements": num_persistent_blocks * hc_mult * hidden, "role": "output", "eps": 2e-2},
                },
                "shape": {
                    "num_tokens": num_tokens,
                    "num_persistent_blocks": num_persistent_blocks,
                    "hc_mult": hc_mult,
                    "hidden": hidden,
                    "clamp_value": clamp_value,
                },
            })
        return specs

    if name in {"mhc.expand_to_mhc_fwd", "mhc.expand_to_mhc_bwd"}:
        mhc_mult = int(config["mhc_mult"])
        specs = []
        for n0 in (1, 2):
            for n1 in (1024, 4096):
                for hidden in (1280, 2560, 7168):
                    tokens = n0 * n1
                    if name.endswith("_fwd"):
                        buffers = {
                            "x": {"dtype": "bf16", "elements": tokens * hidden, "role": "input"},
                            "out": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "output", "eps": 0.0},
                        }
                    else:
                        buffers = {
                            "out_grad": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "input"},
                            "x_grad": {"dtype": "bf16", "elements": tokens * hidden, "role": "output", "eps": 1e-2},
                        }
                    specs.append({
                        "id": _spec_id({"hidden": hidden, "mhc_mult": mhc_mult, "n0": n0, "n1": n1}),
                        "case": name,
                        "config": config,
                        "seed": 401 + mhc_mult + n0 + n1 + hidden + (100 if name.endswith("_bwd") else 0),
                        "scalars": {"tokens": tokens, "hidden": hidden},
                        "buffers": buffers,
                        "shape": {"n0": n0, "n1": n1, "tokens": tokens, "mhc_mult": mhc_mult, "hidden": hidden},
                    })
        return specs

    if name in {"mhc.pre_apply_mix_fwd", "mhc.pre_apply_mix_bwd"}:
        mhc_mult = int(config["mhc_mult"])
        specs = []
        for n0 in (1, 2):
            for n1 in (1024, 4096):
                for hidden in (1280, 2560, 7680):
                    tokens = n0 * n1
                    if name.endswith("_fwd"):
                        buffers = {
                            "x": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "input"},
                            "mix": {"dtype": "f32", "elements": tokens * mhc_mult, "role": "input"},
                            "out": {"dtype": "bf16", "elements": tokens * hidden, "role": "output", "eps": 1e-2},
                        }
                    else:
                        buffers = {
                            "out_grad": {"dtype": "bf16", "elements": tokens * hidden, "role": "input"},
                            "x": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "input"},
                            "mix": {"dtype": "f32", "elements": tokens * mhc_mult, "role": "input"},
                            "x_grad": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "inout", "eps": 1e-2},
                            "mix_grad": {"dtype": "f32", "elements": tokens * mhc_mult, "role": "output", "eps": 2e-2},
                        }
                    specs.append({
                        "id": _spec_id({"hidden": hidden, "mhc_mult": mhc_mult, "n0": n0, "n1": n1}),
                        "case": name,
                        "config": config,
                        "seed": 421 + mhc_mult + n0 + n1 + hidden + (100 if name.endswith("_bwd") else 0),
                        "scalars": {"tokens": tokens, "hidden": hidden},
                        "buffers": buffers,
                        "shape": {"n0": n0, "n1": n1, "tokens": tokens, "mhc_mult": mhc_mult, "hidden": hidden},
                    })
        return specs

    if name in {"mhc.post_fwd", "mhc.post_bwd"}:
        mhc_mult = int(config["mhc_mult"])
        specs = []
        for n0 in (1, 2):
            for n1 in (4096,):
                for hidden in (1280, 2560, 7168):
                    tokens = n0 * n1
                    if name.endswith("_fwd"):
                        buffers = {
                            "comb_res_mix": {"dtype": "f32", "elements": tokens * mhc_mult * mhc_mult, "role": "input"},
                            "residual": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "input"},
                            "post_layer_mix": {"dtype": "f32", "elements": tokens * mhc_mult, "role": "input"},
                            "x": {"dtype": "bf16", "elements": tokens * hidden, "role": "input"},
                            "out": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "output", "eps": 2e-2},
                        }
                    else:
                        buffers = {
                            "out_grad": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "input"},
                            "comb_res_mix": {"dtype": "f32", "elements": tokens * mhc_mult * mhc_mult, "role": "input"},
                            "residual": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "input"},
                            "post_layer_mix": {"dtype": "f32", "elements": tokens * mhc_mult, "role": "input"},
                            "x": {"dtype": "bf16", "elements": tokens * hidden, "role": "input"},
                            "comb_res_mix_grad": {"dtype": "f32", "elements": tokens * mhc_mult * mhc_mult, "role": "output", "eps": 3e-2},
                            "residual_grad": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "output", "eps": 3e-2},
                            "post_layer_mix_grad": {"dtype": "f32", "elements": tokens * mhc_mult, "role": "output", "eps": 3e-2},
                            "x_grad": {"dtype": "bf16", "elements": tokens * hidden, "role": "output", "eps": 3e-2},
                        }
                    specs.append({
                        "id": _spec_id({"hidden": hidden, "mhc_mult": mhc_mult, "n0": n0, "n1": n1}),
                        "case": name,
                        "config": config,
                        "seed": 441 + mhc_mult + n0 + n1 + hidden + (100 if name.endswith("_bwd") else 0),
                        "scalars": {"tokens": tokens, "hidden": hidden},
                        "buffers": buffers,
                        "shape": {"n0": n0, "n1": n1, "tokens": tokens, "mhc_mult": mhc_mult, "hidden": hidden},
                    })
        return specs

    if name == "mhc.pre_norm_fn_fwd":
        hidden = int(config["hidden_size"])
        mhc_mult = int(config["mhc_mult"])
        eps = float(config["eps"])
        mhc_mult3 = mhc_mult * (2 + mhc_mult)
        specs = []
        for n1 in (13, 4096):
            tokens = n1
            specs.append({
                "id": _spec_id({"hidden": hidden, "mhc_mult": mhc_mult, "tokens": tokens}),
                "case": name,
                "config": config,
                "seed": 461 + hidden + mhc_mult + tokens,
                "scalars": {"num_tokens": tokens},
                "buffers": {
                    "residual": {"dtype": "bf16", "elements": tokens * mhc_mult * hidden, "role": "input"},
                    "mhc_fn": {"dtype": "f32", "elements": mhc_mult3 * mhc_mult * hidden, "role": "input"},
                    "output": {"dtype": "f32", "elements": tokens * mhc_mult3, "role": "output", "eps": 2e-2},
                },
                "shape": {
                    "num_tokens": tokens,
                    "mhc_mult": mhc_mult,
                    "mhc_mult3": mhc_mult3,
                    "hidden": hidden,
                    "eps": eps,
                },
            })
        return specs

    if name in {"mhc.sinkhorn_normalize_fwd", "mhc.sinkhorn_normalize_bwd"}:
        mhc = int(config["mhc"])
        repeat = int(config["repeat"])
        eps = float(config["eps"])
        specs = []
        for n0 in (1, 2):
            for n1 in (1, 1024, 4096):
                tokens = n0 * n1
                if name.endswith("_fwd"):
                    buffers = {
                        "x": {"dtype": "f32", "elements": tokens * mhc * mhc, "role": "input"},
                        "out": {"dtype": "f32", "elements": tokens * mhc * mhc, "role": "output", "eps": 1e-5},
                    }
                else:
                    buffers = {
                        "grad_output": {"dtype": "f32", "elements": tokens * mhc * mhc, "role": "input"},
                        "x": {"dtype": "f32", "elements": tokens * mhc * mhc, "role": "input"},
                        "grad_input": {"dtype": "f32", "elements": tokens * mhc * mhc, "role": "output", "eps": 1e-4},
                    }
                specs.append({
                    "id": _spec_id({"mhc": mhc, "n0": n0, "n1": n1, "repeat": repeat}),
                    "case": name,
                    "config": config,
                    "seed": 601 + n0 + n1 + mhc + repeat + (100 if name.endswith("_bwd") else 0),
                    "scalars": {"num_tokens": tokens},
                    "buffers": buffers,
                    "shape": {"n0": n0, "n1": n1, "num_tokens": tokens, "mhc": mhc, "repeat": repeat, "eps": eps},
                })
        return specs

    if name in {"mhc.head_compute_mix_fwd", "mhc.head_compute_mix_bwd"}:
        mhc_mult = int(config["mhc_mult"])
        eps = float(config.get("eps", 0.0))
        specs = []
        for num_tokens in (0, 4001):
            if name.endswith("_fwd"):
                buffers = {
                    "input_mix": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "input"},
                    "mhc_scale": {"dtype": "f32", "elements": 1, "role": "input"},
                    "mhc_base": {"dtype": "f32", "elements": mhc_mult, "role": "input"},
                    "output_mix": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "output", "eps": 1e-5},
                }
            else:
                buffers = {
                    "output_mix_grad": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "input"},
                    "input_mix": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "input"},
                    "mhc_scale": {"dtype": "f32", "elements": 1, "role": "input"},
                    "mhc_base": {"dtype": "f32", "elements": mhc_mult, "role": "input"},
                    "input_mix_grad": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "output", "eps": 1e-5},
                    "mhc_scale_grad_partial": {"dtype": "f32", "elements": 1, "role": "output", "eps": 1e-3},
                    "mhc_base_grad_partial": {"dtype": "f32", "elements": mhc_mult, "role": "output", "eps": 1e-3},
                }
            specs.append({
                "id": _spec_id({"mhc_mult": mhc_mult, "tokens": num_tokens, "dir": "fwd" if name.endswith("_fwd") else "bwd"}),
                "case": name,
                "config": config,
                "seed": 641 + mhc_mult + num_tokens + (100 if name.endswith("_bwd") else 0),
                "scalars": {"num_tokens": num_tokens},
                "buffers": buffers,
                "shape": {
                    "num_tokens": num_tokens,
                    "mhc_mult": mhc_mult,
                    "eps": eps,
                },
            })
        return specs

    if name in {"mhc.pre_split_mixes_fwd", "mhc.pre_split_mixes_bwd"}:
        mhc_mult = int(config["mhc_mult"])
        mhc_post_mult_value = float(config["mhc_post_mult_value"])
        mhc_pre_eps = float(config.get("mhc_pre_eps", 0.0))
        mhc_mult2 = mhc_mult * mhc_mult
        mhc_mult3 = mhc_mult * 2 + mhc_mult2
        specs = []
        for num_tokens in (0, 4001):
            if name.endswith("_fwd"):
                buffers = {
                    "input_mixes": {"dtype": "f32", "elements": num_tokens * mhc_mult3, "role": "input"},
                    "mhc_scale": {"dtype": "f32", "elements": 3, "role": "input"},
                    "mhc_base": {"dtype": "f32", "elements": mhc_mult3, "role": "input"},
                    "pre_layer_mix": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "output", "eps": 1e-5},
                    "post_layer_mix": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "output", "eps": 1e-5},
                    "comb_res_mix": {"dtype": "f32", "elements": num_tokens * mhc_mult2, "role": "output", "eps": 1e-5},
                }
            else:
                buffers = {
                    "pre_layer_mix_grad": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "input"},
                    "post_layer_mix_grad": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "input"},
                    "comb_res_mix_grad": {"dtype": "f32", "elements": num_tokens * mhc_mult2, "role": "input"},
                    "input_mixes": {"dtype": "f32", "elements": num_tokens * mhc_mult3, "role": "input"},
                    "post_layer_mix": {"dtype": "f32", "elements": num_tokens * mhc_mult, "role": "input"},
                    "mhc_scale": {"dtype": "f32", "elements": 3, "role": "input"},
                    "mhc_base": {"dtype": "f32", "elements": mhc_mult3, "role": "input"},
                    "input_mixes_grad": {"dtype": "f32", "elements": num_tokens * mhc_mult3, "role": "output", "eps": 1e-5},
                    "mhc_scale_grad_partial": {"dtype": "f32", "elements": 3, "role": "output", "eps": 1e-3},
                    "mhc_base_grad_partial": {"dtype": "f32", "elements": mhc_mult3, "role": "output", "eps": 1e-3},
                }
            specs.append({
                "id": _spec_id({"mhc_mult": mhc_mult, "tokens": num_tokens, "dir": "fwd" if name.endswith("_fwd") else "bwd"}),
                "case": name,
                "config": config,
                "seed": 681 + mhc_mult + num_tokens + (100 if name.endswith("_bwd") else 0),
                "scalars": {"num_tokens": num_tokens},
                "buffers": buffers,
                "shape": {
                    "num_tokens": num_tokens,
                    "mhc_mult": mhc_mult,
                    "mhc_mult2": mhc_mult2,
                    "mhc_mult3": mhc_mult3,
                    "mhc_post_mult_value": mhc_post_mult_value,
                    "mhc_pre_eps": mhc_pre_eps,
                },
            })
        return specs

    raise ValueError(f"no deterministic validation spec registered for {name}")


def _arg_files(case) -> dict[str, str]:
    return {arg.name: f"v{idx}.bin" for idx, arg in enumerate(case.args, start=1)}


def _golden_files(case) -> dict[str, str]:
    return {arg.name: f"golden_v{idx}.bin" for idx, arg in enumerate(case.args, start=1)}


def _write_generated_main(case, config: dict[str, Any], symbol: str, spec: dict[str, Any]) -> str:
    wrapper = f"Launch_{symbol}"
    ptr_params = ", ".join(_ptr_decl(arg, config) for arg in case.args)
    call_args = ", ".join(_host_call_arg(arg, config) for arg in case.args)
    files = _arg_files(case)

    pointer_args = [arg for arg in case.args if arg.pointer]
    scalar_args = [arg for arg in case.args if not arg.pointer]
    outputs = {
        name
        for name, buf in spec["buffers"].items()
        if buf.get("role") in {"output", "inout"}
    }

    declarations: list[str] = []
    allocations: list[str] = []
    reads: list[str] = []
    h2d: list[str] = []
    d2h: list[str] = []
    writes: list[str] = []
    cleanup: list[str] = []

    for arg in pointer_args:
        buf = spec["buffers"][arg.name]
        nbytes = int(buf["elements"]) * _dtype_size(str(buf["dtype"]))
        declarations.extend(
            [
                f"    constexpr size_t {arg.name}_bytes = {nbytes};",
                f"    void *{arg.name}_host = nullptr;",
                f"    void *{arg.name}_device = nullptr;",
                f"    size_t {arg.name}_read_size = {arg.name}_bytes;",
            ]
        )
        if nbytes > 0:
            allocations.extend(
                [
                    f"    ACL_CHECK(aclrtMallocHost(&{arg.name}_host, {arg.name}_bytes));",
                    f"    ACL_CHECK(aclrtMalloc(&{arg.name}_device, {arg.name}_bytes, ACL_MEM_MALLOC_HUGE_FIRST));",
                ]
            )
            reads.append(
                f"    FILE_CHECK(ReadFile(\"./{files[arg.name]}\", {arg.name}_read_size, {arg.name}_host, {arg.name}_bytes));"
            )
            h2d.append(
                f"    ACL_CHECK(aclrtMemcpy({arg.name}_device, {arg.name}_bytes, {arg.name}_host, {arg.name}_bytes, ACL_MEMCPY_HOST_TO_DEVICE));"
            )
            if arg.name in outputs:
                d2h.append(
                    f"    ACL_CHECK(aclrtMemcpy({arg.name}_host, {arg.name}_bytes, {arg.name}_device, {arg.name}_bytes, ACL_MEMCPY_DEVICE_TO_HOST));"
                )
                writes.append(
                    f"    FILE_CHECK(WriteFile(\"./{files[arg.name]}\", {arg.name}_host, {arg.name}_bytes));"
                )
            cleanup.extend(
                [
                    f"    if ({arg.name}_device != nullptr) aclrtFree({arg.name}_device);",
                    f"    if ({arg.name}_host != nullptr) aclrtFreeHost({arg.name}_host);",
                ]
            )

    for arg in scalar_args:
        scalar_value = int(spec["scalars"][arg.name])
        declarations.extend(
            [
                f"    int32_t {arg.name}_host = {scalar_value};",
                f"    size_t {arg.name}_read_size = sizeof({arg.name}_host);",
            ]
        )
        reads.append(
            f"    FILE_CHECK(ReadFile(\"./{files[arg.name]}\", {arg.name}_read_size, &{arg.name}_host, sizeof({arg.name}_host)));"
        )

    return f"""#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#define ACL_CHECK(expr) \\
    do {{ \\
        const aclError _ret = (expr); \\
        if (_ret != ACL_SUCCESS) {{ \\
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\\n", #expr, (int)_ret, __FILE__, __LINE__); \\
            const char *_recent = aclGetRecentErrMsg(); \\
            if (_recent != nullptr && _recent[0] != '\\0') \\
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\\n", _recent); \\
            rc = 1; \\
            goto cleanup; \\
        }} \\
    }} while (0)

#define FILE_CHECK(expr) \\
    do {{ \\
        if (!(expr)) {{ \\
            std::fprintf(stderr, "[ERROR] %s failed (%s:%d)\\n", #expr, __FILE__, __LINE__); \\
            rc = 1; \\
            goto cleanup; \\
        }} \\
    }} while (0)

void {wrapper}({ptr_params}, void *stream);

int main() {{
{chr(10).join(declarations)}
    int rc = 0;
    bool acl_inited = false;
    bool device_set = false;
    int device_id = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    acl_inited = true;
    if (const char *env_device = std::getenv("ACL_DEVICE_ID"))
        device_id = std::atoi(env_device);
    ACL_CHECK(aclrtSetDevice(device_id));
    device_set = true;
    ACL_CHECK(aclrtCreateStream(&stream));

{chr(10).join(allocations)}
{chr(10).join(reads)}
{chr(10).join(h2d)}

    {wrapper}({call_args}, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

{chr(10).join(d2h)}
{chr(10).join(writes)}

cleanup:
{chr(10).join(cleanup)}
    if (stream != nullptr) {{
        const aclError _ret = aclrtDestroyStream(stream);
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] aclrtDestroyStream failed: %d\\n", (int)_ret);
    }}
    if (device_set) {{
        const aclError _ret = aclrtResetDevice(device_id);
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] aclrtResetDevice failed: %d\\n", (int)_ret);
    }}
    if (acl_inited) {{
        const aclError _ret = aclFinalize();
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] aclFinalize failed: %d\\n", (int)_ret);
    }}
    return rc;
}}
"""


def _write_generated_golden(case, spec: dict[str, Any]) -> str:
    files = _arg_files(case)
    golden_files = _golden_files(case)
    payload = json.dumps({"spec": spec, "files": files, "golden_files": golden_files}, indent=2, sort_keys=True)
    return f"""#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


CASE = json.loads({payload!r})


def f32_to_bf16_bits(values):
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_f32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def write_array(path: Path, dtype: str, values) -> None:
    if dtype == "f32":
        np.asarray(values, dtype=np.float32).reshape(-1).tofile(path)
    elif dtype == "bf16":
        f32_to_bf16_bits(values).reshape(-1).tofile(path)
    elif dtype == "i32":
        np.asarray(values, dtype=np.int32).reshape(-1).tofile(path)
    elif dtype == "i64":
        np.asarray(values, dtype=np.int64).reshape(-1).tofile(path)
    elif dtype == "u32":
        np.asarray(values, dtype=np.uint32).reshape(-1).tofile(path)
    else:
        raise ValueError(f"unsupported dtype {{dtype}}")


def zeros(dtype: str, elements: int):
    if dtype == "f32":
        return np.zeros(elements, dtype=np.float32)
    if dtype == "bf16":
        return np.zeros(elements, dtype=np.float32)
    if dtype == "i32":
        return np.zeros(elements, dtype=np.int32)
    if dtype == "i64":
        return np.zeros(elements, dtype=np.int64)
    if dtype == "u32":
        return np.zeros(elements, dtype=np.uint32)
    raise ValueError(f"unsupported zero dtype {{dtype}}")


def stable_topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :k].astype(np.int64)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-values.astype(np.float32)))).astype(np.float32)


def sinkhorn_forward_np(values: np.ndarray, repeat: int, eps: float) -> np.ndarray:
    x = values.astype(np.float32)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True)).astype(np.float32)
    x = (x / np.sum(x, axis=-1, keepdims=True)).astype(np.float32)
    x = (x + np.float32(eps)).astype(np.float32)
    x = (x / (np.sum(x, axis=-2, keepdims=True) + np.float32(eps))).astype(np.float32)
    for _ in range(repeat - 1):
        x = (x / (np.sum(x, axis=-1, keepdims=True) + np.float32(eps))).astype(np.float32)
        x = (x / (np.sum(x, axis=-2, keepdims=True) + np.float32(eps))).astype(np.float32)
    return x


def sinkhorn_backward_np(grad_output: np.ndarray, values: np.ndarray, repeat: int, eps: float) -> np.ndarray:
    x = values.astype(np.float32)
    grad = grad_output.astype(np.float32)
    stages = []
    sums = []

    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True)).astype(np.float32)
    x_softmax = (x_exp / np.sum(x_exp, axis=-1, keepdims=True)).astype(np.float32)
    stages.append(x_softmax)
    sums.append(np.zeros(x_softmax.shape[:-1], dtype=np.float32))

    x = (x_softmax + np.float32(eps)).astype(np.float32)
    stages.append(x)
    col_sum = np.sum(x, axis=-2, dtype=np.float32)
    sums.append(col_sum)
    x = (x / (col_sum[..., None, :] + np.float32(eps))).astype(np.float32)

    for _ in range(repeat - 1):
        stages.append(x)
        row_sum = np.sum(x, axis=-1, dtype=np.float32)
        sums.append(row_sum)
        x = (x / (row_sum[..., :, None] + np.float32(eps))).astype(np.float32)

        stages.append(x)
        col_sum = np.sum(x, axis=-2, dtype=np.float32)
        sums.append(col_sum)
        x = (x / (col_sum[..., None, :] + np.float32(eps))).astype(np.float32)

    for inv_step in range(2 * repeat - 1):
        stage_idx = 2 * repeat - 1 - inv_step
        x_inter = stages[stage_idx]
        norm_sum = sums[stage_idx]
        if inv_step % 2 == 0:
            temp = np.sum(grad * x_inter, axis=-2, dtype=np.float32)
            temp = (temp / (norm_sum + np.float32(eps))).astype(np.float32)
            grad = ((grad - temp[..., None, :]) / (norm_sum[..., None, :] + np.float32(eps))).astype(np.float32)
        else:
            temp = np.sum(grad * x_inter, axis=-1, dtype=np.float32)
            temp = (temp / (norm_sum + np.float32(eps))).astype(np.float32)
            grad = ((grad - temp[..., :, None]) / (norm_sum[..., :, None] + np.float32(eps))).astype(np.float32)

    softmax = stages[0]
    temp = np.sum(grad * softmax, axis=-1, dtype=np.float32)
    return ((grad - temp[..., :, None]) * softmax).astype(np.float32)


def generate(output_dir: Path) -> None:
    spec = CASE["spec"]
    files = CASE["files"]
    golden_files = CASE["golden_files"]
    rng = np.random.default_rng(int(spec["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, value in spec["scalars"].items():
        write_array(output_dir / files[name], "i32", [value])

    case = spec["case"]
    shape = spec["shape"]
    generated = {{}}
    golden = {{}}

    if case == "moe.normalize_weight":
        rows = int(shape["num_tokens"])
        cols = int(shape["num_topk"])
        topk_weights = rng.uniform(0.05, 1.0, size=(rows, cols)).astype(np.float32)
        denominator = np.float32(1e-20) + np.sum(topk_weights, axis=1, dtype=np.float32)
        normalized_weights = topk_weights / denominator[:, None]
        generated["topk_weights"] = topk_weights
        golden["denominator"] = denominator
        golden["normalized_weights"] = normalized_weights
    elif case == "moe.mask_indices_by_tp":
        rows = int(shape["num_tokens"])
        cols = int(shape["num_topk"])
        n = int(shape["n"])
        per_gpu = int(spec["scalars"]["per_gpu"])
        per_dp = int(spec["scalars"]["per_dp"])
        num_tp_ranks = int(spec["scalars"]["num_tp_ranks"])
        tp_rank = int(spec["scalars"]["tp_rank"])
        indices = rng.integers(-1, n, size=(rows, cols), dtype=np.int64)
        negative_mask = rng.random((rows, cols)) < 0.08
        indices[negative_mask] = -1
        masked = np.full_like(indices, -1)
        valid = indices >= 0
        rank = (indices // per_gpu) % num_tp_ranks
        local_value = indices - tp_rank * per_gpu
        dp_rank = local_value // per_dp
        remapped = local_value - dp_rank * (per_dp - per_gpu)
        keep = valid & (rank == tp_rank) & (remapped >= 0)
        masked[keep] = remapped[keep]
        generated["indices"] = indices
        golden["masked_indices"] = masked
    elif case == "moe.group_count":
        rows = int(shape["num_tokens"])
        cols = int(shape["num_topk"])
        num_groups = int(shape["num_groups"])
        group_idx = rng.integers(-1, num_groups, size=(rows, cols), dtype=np.int64)
        negative_mask = rng.random((rows, cols)) < 0.07
        group_idx[negative_mask] = -1
        out = np.zeros((num_groups,), dtype=np.int32)
        for expert in range(num_groups):
            out[expert] = np.count_nonzero(group_idx == expert)
        generated["group_idx"] = group_idx
        golden["out"] = out
    elif case == "moe.aux_fi":
        rows = int(shape["num_tokens"])
        cols = int(shape["num_topk"])
        num_experts = int(shape["num_experts"])
        num_aux_topk = int(shape["num_aux_topk"])
        topk_idx = rng.integers(-1, num_experts, size=(rows, cols), dtype=np.int64)
        negative_mask = rng.random((rows, cols)) < 0.07
        topk_idx[negative_mask] = -1
        out = np.zeros((num_experts,), dtype=np.float32)
        denom = np.float32(rows * num_aux_topk)
        for expert in range(num_experts):
            out[expert] = np.float32(np.count_nonzero(topk_idx == expert) * num_experts) / denom
        generated["topk_idx"] = topk_idx
        golden["out"] = out
    elif case == "moe.topk_gate":
        rows = int(shape["num_tokens"])
        num_experts = int(shape["num_experts"])
        num_topk = int(shape["num_topk"])
        sort_ncols = int(shape["sort_ncols"])
        scores = rng.normal(0.0, 1.0, size=(rows, num_experts)).astype(np.float32)
        if rows > 0:
            scores[0, :] = 0.0
        generated["scores"] = scores
        generated["inidx"] = np.arange(sort_ncols, dtype=np.uint32)
        golden["topk_idx"] = stable_topk_indices(scores, num_topk).astype(np.uint32)
    elif case == "moe.topk_sum_and_topk_group_idx":
        rows = int(shape["num_tokens"])
        num_experts = int(shape["num_experts"])
        num_groups = int(shape["num_groups"])
        experts_per_group = int(shape["experts_per_group"])
        num_group_sum_topk = int(shape["num_group_sum_topk"])
        num_topk_groups = int(shape["num_topk_groups"])
        scores = rng.normal(0.0, 1.0, size=(rows, num_experts)).astype(np.float32)
        if rows > 0:
            scores[0, :] = 0.0
        grouped = scores.reshape(rows, num_groups, experts_per_group)
        group_scores = np.zeros((rows, num_groups), dtype=np.float32)
        for group in range(num_groups):
            idx = stable_topk_indices(grouped[:, group, :], num_group_sum_topk)
            group_scores[:, group] = np.take_along_axis(grouped[:, group, :], idx, axis=1).sum(axis=1, dtype=np.float32)
        generated["scores"] = scores
        generated["inidx"] = np.arange(int(shape["max_sort_ncols"]), dtype=np.uint32)
        golden["group_idx"] = stable_topk_indices(group_scores, num_topk_groups).astype(np.uint32)
    elif case == "moe.inplace_unique_group_indices":
        rows = int(shape["num_tokens"])
        cols = int(shape["num_topk"])
        num_groups = int(shape["num_groups"])
        group_indices = rng.integers(-1, num_groups, size=(rows, cols), dtype=np.int64)
        if rows > 0 and cols > 1:
            group_indices[0, :] = 0
            group_indices[1::5, min(1, cols - 1)] = group_indices[1::5, 0]
        deduped = group_indices.copy()
        for row in range(rows):
            seen = set()
            for col in range(cols):
                value = int(deduped[row, col])
                if value < 0:
                    continue
                if value in seen:
                    deduped[row, col] = -1
                else:
                    seen.add(value)
        generated["group_indices"] = group_indices
        golden["group_indices"] = deduped
    elif case == "transpose.transpose":
        rows = int(shape["rows"])
        cols = int(shape["cols"])
        x = rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
        generated["x"] = x
        golden["out"] = x.T
    elif case == "transpose.batched_transpose":
        batches = int(shape["batches"])
        rows = int(shape["rows"])
        cols = int(shape["cols"])
        x = rng.normal(0.0, 1.0, size=(batches, rows, cols)).astype(np.float32)
        generated["x"] = x
        golden["out"] = np.transpose(x, (0, 2, 1))
    elif case == "engram.fused_weight":
        hc = int(shape["hc"])
        hidden = int(shape["hidden"])
        weight_hidden = rng.normal(0.0, 0.75, size=(hc, hidden)).astype(np.float32)
        weight_embed = rng.normal(0.0, 0.75, size=(hc, hidden)).astype(np.float32)
        wh = bf16_bits_to_f32(f32_to_bf16_bits(weight_hidden))
        we = bf16_bits_to_f32(f32_to_bf16_bits(weight_embed))
        generated["weight_hidden"] = weight_hidden
        generated["weight_embed"] = weight_embed
        golden["weight_fused"] = wh * we
    elif case == "engram.engram_hash":
        rows = int(shape["num_tokens"])
        max_ngram_size = int(shape["max_ngram_size"])
        num_ngram_layers = int(shape["num_ngram_layers"])
        num_embed_table_per_ngram = int(shape["num_embed_table_per_ngram"])
        num_out_cols = int(shape["num_out_cols"])
        ngram_token_ids = rng.integers(0, 100000, size=(rows, max_ngram_size), dtype=np.int32)
        multipliers = rng.integers(1, 100000, size=(num_ngram_layers, max_ngram_size), dtype=np.int64)
        vocab_sizes = rng.integers(
            100000,
            1000000,
            size=(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram),
            dtype=np.int32,
        )
        offsets = np.zeros((num_ngram_layers, num_out_cols), dtype=np.int32)
        for layer in range(num_ngram_layers):
            running = 0
            for col in range(num_out_cols):
                offsets[layer, col] = running
                running += int(vocab_sizes[layer, col // num_embed_table_per_ngram, col % num_embed_table_per_ngram])
        output = np.zeros((num_ngram_layers, rows, num_out_cols), dtype=np.int32)
        for layer in range(num_ngram_layers):
            for token in range(rows):
                hash_value = np.int64(0)
                for ngram_idx in range(max_ngram_size):
                    hash_value = np.bitwise_xor(
                        hash_value,
                        np.int64(ngram_token_ids[token, ngram_idx]) * multipliers[layer, ngram_idx],
                    )
                    if ngram_idx > 0:
                        for table in range(num_embed_table_per_ngram):
                            col = (ngram_idx - 1) * num_embed_table_per_ngram + table
                            output[layer, token, col] = (
                                int(hash_value % np.int64(vocab_sizes[layer, ngram_idx - 1, table]))
                                + int(offsets[layer, col])
                            )
        generated["ngram_token_ids"] = ngram_token_ids
        generated["multipliers"] = multipliers
        generated["vocab_sizes"] = vocab_sizes
        generated["offsets"] = offsets
        golden["output"] = output
    elif case == "engram.grad_w_reduce":
        num_persistent_blocks = int(shape["num_persistent_blocks"])
        hc = int(shape["hc_mult"])
        hidden = int(shape["hidden"])
        grad_w_partial = rng.normal(
            0.0, 0.25, size=(num_persistent_blocks, hc, hidden)
        ).astype(np.float32)
        weight_hidden = rng.normal(0.0, 0.75, size=(hc, hidden)).astype(np.float32)
        weight_embed = rng.normal(0.0, 0.75, size=(hc, hidden)).astype(np.float32)
        grad_weight_hidden = rng.normal(0.0, 0.1, size=(hc, hidden)).astype(np.float32)
        grad_weight_embed = rng.normal(0.0, 0.1, size=(hc, hidden)).astype(np.float32)
        wh = bf16_bits_to_f32(f32_to_bf16_bits(weight_hidden))
        we = bf16_bits_to_f32(f32_to_bf16_bits(weight_embed))
        grad_w_sum = np.sum(grad_w_partial, axis=0, dtype=np.float32)
        generated["grad_w_partial"] = grad_w_partial
        generated["weight_hidden"] = weight_hidden
        generated["weight_embed"] = weight_embed
        generated["grad_weight_hidden"] = grad_weight_hidden
        generated["grad_weight_embed"] = grad_weight_embed
        golden["grad_weight_hidden"] = grad_weight_hidden + grad_w_sum * we
        golden["grad_weight_embed"] = grad_weight_embed + grad_w_sum * wh
    elif case == "engram.engram_gate_fwd":
        rows = int(shape["num_tokens"])
        hc = int(shape["hc_mult"])
        hidden = int(shape["hidden"])
        eps = np.float32(float(shape["eps"]))
        clamp_value = np.float32(float(shape["clamp_value"]))
        hidden_states = rng.normal(0.0, 1.0, size=(rows, hc, hidden)).astype(np.float32)
        k = rng.normal(0.0, 1.0, size=(rows, hc, hidden)).astype(np.float32)
        v = rng.normal(0.0, 1.0, size=(rows, hidden)).astype(np.float32)
        weight_fused = rng.normal(0.0, 0.5, size=(hc, hidden)).astype(np.float32)
        x = bf16_bits_to_f32(f32_to_bf16_bits(hidden_states))
        k_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(k))
        v_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(v))
        raw_dot = np.sum(x * k_bf16 * weight_fused[None, :, :], axis=-1, dtype=np.float32)
        rstd_x = np.float32(1.0) / np.sqrt(
            np.mean(x * x, axis=-1, dtype=np.float32) + eps
        ).astype(np.float32)
        rstd_k = np.float32(1.0) / np.sqrt(
            np.mean(k_bf16 * k_bf16, axis=-1, dtype=np.float32) + eps
        ).astype(np.float32)
        dot = raw_dot * rstd_x * rstd_k * np.float32(hidden ** -0.5)
        signed_sqrt = np.sqrt(
            np.maximum(np.abs(dot), clamp_value)
        ).astype(np.float32) * np.sign(dot).astype(np.float32)
        gate_score = sigmoid_np(signed_sqrt)
        output = x + gate_score[:, :, None] * v_bf16[:, None, :]
        generated["hidden_states"] = hidden_states
        generated["k"] = k
        generated["v"] = v
        generated["weight_fused"] = weight_fused
        golden["output"] = output
        golden["dot_out"] = raw_dot
        golden["gate_score"] = gate_score
        golden["rstd_x"] = rstd_x
        golden["rstd_k"] = rstd_k
    elif case == "engram.engram_gate_bwd":
        rows = int(shape["num_tokens"])
        hc = int(shape["hc_mult"])
        hidden = int(shape["hidden"])
        num_persistent_blocks = int(shape["num_persistent_blocks"])
        clamp_value = np.float32(float(shape["clamp_value"]))
        hidden_states = rng.normal(0.0, 1.0, size=(rows, hc, hidden)).astype(np.float32)
        k = rng.normal(0.0, 1.0, size=(rows, hc, hidden)).astype(np.float32)
        v = rng.normal(0.0, 1.0, size=(rows, hidden)).astype(np.float32)
        weight_fused = rng.normal(0.0, 0.5, size=(hc, hidden)).astype(np.float32)
        grad_out = rng.normal(0.0, 0.25, size=(rows, hc, hidden)).astype(np.float32)
        x = bf16_bits_to_f32(f32_to_bf16_bits(hidden_states))
        k_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(k))
        v_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(v))
        go = bf16_bits_to_f32(f32_to_bf16_bits(grad_out))
        raw_dot = np.sum(x * k_bf16 * weight_fused[None, :, :], axis=-1, dtype=np.float32)
        rstd_x = np.float32(1.0) / np.sqrt(
            np.mean(x * x, axis=-1, dtype=np.float32) + np.float32(1e-20)
        ).astype(np.float32)
        rstd_k = np.float32(1.0) / np.sqrt(
            np.mean(k_bf16 * k_bf16, axis=-1, dtype=np.float32) + np.float32(1e-20)
        ).astype(np.float32)
        scale = np.float32(hidden ** -0.5)
        dot = raw_dot * rstd_x * rstd_k * scale
        signed_sqrt = np.sqrt(
            np.maximum(np.abs(dot), clamp_value)
        ).astype(np.float32) * np.sign(dot).astype(np.float32)
        gate = sigmoid_np(signed_sqrt)
        dldg = np.sum(go * v_bf16[:, None, :], axis=-1, dtype=np.float32)
        coeff = rstd_x * rstd_k * scale
        abs_dot = np.abs(raw_dot)
        active = abs_dot * coeff >= clamp_value
        denom = np.maximum(abs_dot, np.float32(1e-20))
        dldg_r = (
            dldg
            * gate
            * (np.float32(1.0) - gate)
            * np.float32(0.5)
            * np.sqrt(coeff / denom).astype(np.float32)
        )
        dldg_r = np.where(active, dldg_r, np.float32(0.0)).astype(np.float32)
        dot_x = raw_dot * rstd_x * rstd_x / np.float32(hidden)
        dot_k = raw_dot * rstd_k * rstd_k / np.float32(hidden)
        grad_x = go + dldg_r[:, :, None] * (
            k_bf16 * weight_fused[None, :, :] - x * dot_x[:, :, None]
        )
        grad_k = dldg_r[:, :, None] * (
            x * weight_fused[None, :, :] - k_bf16 * dot_k[:, :, None]
        )
        grad_v = np.sum(go * gate[:, :, None], axis=1, dtype=np.float32)
        grad_w_partial = np.zeros((num_persistent_blocks, hc, hidden), dtype=np.float32)
        rows_per_block = (rows + num_persistent_blocks - 1) // num_persistent_blocks
        for block_idx in range(num_persistent_blocks):
            start = min(block_idx * rows_per_block, rows)
            end = min(start + rows_per_block, rows)
            if start < end:
                grad_w_partial[block_idx] = np.sum(
                    dldg_r[start:end, :, None] * x[start:end] * k_bf16[start:end],
                    axis=0,
                    dtype=np.float32,
                )
        generated["grad_out"] = grad_out
        generated["hidden_states"] = hidden_states
        generated["k"] = k
        generated["v"] = v
        generated["weight_fused"] = weight_fused
        generated["dot_in"] = raw_dot
        generated["gate_in"] = gate
        generated["rstd_x_in"] = rstd_x
        generated["rstd_k_in"] = rstd_k
        golden["grad_x"] = grad_x
        golden["grad_k"] = grad_k
        golden["grad_v"] = grad_v
        golden["grad_w_partial"] = grad_w_partial
    elif case == "mhc.expand_to_mhc_fwd":
        tokens = int(shape["tokens"])
        hidden = int(shape["hidden"])
        mhc_mult = int(shape["mhc_mult"])
        x = rng.normal(0.0, 1.0, size=(tokens, hidden)).astype(np.float32)
        generated["x"] = x
        golden["out"] = np.repeat(x[:, None, :], mhc_mult, axis=1)
    elif case == "mhc.expand_to_mhc_bwd":
        tokens = int(shape["tokens"])
        hidden = int(shape["hidden"])
        mhc_mult = int(shape["mhc_mult"])
        out_grad = rng.normal(0.0, 0.25, size=(tokens, mhc_mult, hidden)).astype(np.float32)
        out_grad_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(out_grad))
        generated["out_grad"] = out_grad
        golden["x_grad"] = np.sum(out_grad_bf16, axis=1, dtype=np.float32)
    elif case == "mhc.pre_apply_mix_fwd":
        tokens = int(shape["tokens"])
        hidden = int(shape["hidden"])
        mhc_mult = int(shape["mhc_mult"])
        x = rng.normal(0.0, 0.75, size=(tokens, mhc_mult, hidden)).astype(np.float32)
        mix_logits = rng.normal(0.0, 0.5, size=(tokens, mhc_mult)).astype(np.float32)
        mix_exp = np.exp(mix_logits - np.max(mix_logits, axis=-1, keepdims=True)).astype(np.float32)
        mix = (mix_exp / np.sum(mix_exp, axis=-1, keepdims=True)).astype(np.float32)
        x_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(x))
        generated["x"] = x
        generated["mix"] = mix
        golden["out"] = np.sum(x_bf16 * mix[:, :, None], axis=1, dtype=np.float32)
    elif case == "mhc.pre_apply_mix_bwd":
        tokens = int(shape["tokens"])
        hidden = int(shape["hidden"])
        mhc_mult = int(shape["mhc_mult"])
        out_grad = rng.normal(0.0, 0.25, size=(tokens, hidden)).astype(np.float32)
        x = rng.normal(0.0, 0.75, size=(tokens, mhc_mult, hidden)).astype(np.float32)
        mix_logits = rng.normal(0.0, 0.5, size=(tokens, mhc_mult)).astype(np.float32)
        mix_exp = np.exp(mix_logits - np.max(mix_logits, axis=-1, keepdims=True)).astype(np.float32)
        mix = (mix_exp / np.sum(mix_exp, axis=-1, keepdims=True)).astype(np.float32)
        x_grad = rng.normal(0.0, 0.05, size=(tokens, mhc_mult, hidden)).astype(np.float32)
        og = bf16_bits_to_f32(f32_to_bf16_bits(out_grad))
        x_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(x))
        xg_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(x_grad))
        generated["out_grad"] = out_grad
        generated["x"] = x
        generated["mix"] = mix
        generated["x_grad"] = x_grad
        golden["x_grad"] = xg_bf16 + mix[:, :, None] * og[:, None, :]
        golden["mix_grad"] = np.sum(og[:, None, :] * x_bf16, axis=-1, dtype=np.float32)
    elif case == "mhc.post_fwd":
        tokens = int(shape["tokens"])
        hidden = int(shape["hidden"])
        mhc = int(shape["mhc_mult"])
        comb_res_mix = rng.normal(0.0, 0.25, size=(tokens, mhc, mhc)).astype(np.float32)
        residual = rng.normal(0.0, 0.75, size=(tokens, mhc, hidden)).astype(np.float32)
        post_layer_mix = rng.normal(0.0, 0.25, size=(tokens, mhc)).astype(np.float32)
        x = rng.normal(0.0, 0.75, size=(tokens, hidden)).astype(np.float32)
        residual_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(residual))
        x_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(x))
        out = (
            x_bf16[:, None, :] * post_layer_mix[:, :, None]
            + np.einsum("tio,tih->toh", comb_res_mix, residual_bf16, dtype=np.float32)
        ).astype(np.float32)
        generated["comb_res_mix"] = comb_res_mix
        generated["residual"] = residual
        generated["post_layer_mix"] = post_layer_mix
        generated["x"] = x
        golden["out"] = out
    elif case == "mhc.post_bwd":
        tokens = int(shape["tokens"])
        hidden = int(shape["hidden"])
        mhc = int(shape["mhc_mult"])
        out_grad = rng.normal(0.0, 0.25, size=(tokens, mhc, hidden)).astype(np.float32)
        comb_res_mix = rng.normal(0.0, 0.25, size=(tokens, mhc, mhc)).astype(np.float32)
        residual = rng.normal(0.0, 0.75, size=(tokens, mhc, hidden)).astype(np.float32)
        post_layer_mix = rng.normal(0.0, 0.25, size=(tokens, mhc)).astype(np.float32)
        x = rng.normal(0.0, 0.75, size=(tokens, hidden)).astype(np.float32)
        og = bf16_bits_to_f32(f32_to_bf16_bits(out_grad))
        residual_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(residual))
        x_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(x))
        generated["out_grad"] = out_grad
        generated["comb_res_mix"] = comb_res_mix
        generated["residual"] = residual
        generated["post_layer_mix"] = post_layer_mix
        generated["x"] = x
        golden["comb_res_mix_grad"] = np.einsum("tih,toh->tio", residual_bf16, og, dtype=np.float32)
        golden["residual_grad"] = np.einsum("tio,toh->tih", comb_res_mix, og, dtype=np.float32)
        golden["post_layer_mix_grad"] = np.sum(x_bf16[:, None, :] * og, axis=-1, dtype=np.float32)
        golden["x_grad"] = np.sum(post_layer_mix[:, :, None] * og, axis=1, dtype=np.float32)
    elif case == "mhc.pre_norm_fn_fwd":
        rows = int(shape["num_tokens"])
        mhc = int(shape["mhc_mult"])
        mhc_mult3 = int(shape["mhc_mult3"])
        hidden = int(shape["hidden"])
        eps = np.float32(float(shape["eps"]))
        residual = rng.normal(0.0, 0.75, size=(rows, mhc, hidden)).astype(np.float32)
        mhc_fn = (
            rng.normal(0.0, 1.0e-4, size=(mhc_mult3, mhc, hidden))
            * (1.0 + np.arange(mhc, dtype=np.float32).reshape(1, mhc, 1) * 0.01)
        ).reshape(mhc_mult3, mhc * hidden).astype(np.float32)
        residual_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(residual)).reshape(rows, mhc * hidden)
        sqsum = np.sum(residual_bf16 * residual_bf16, axis=-1, dtype=np.float32)
        rms = (np.float32(1.0) / np.sqrt(sqsum / np.float32(mhc * hidden) + eps)).astype(np.float32)
        dots = residual_bf16 @ mhc_fn.T
        generated["residual"] = residual
        generated["mhc_fn"] = mhc_fn
        golden["output"] = (dots * rms[:, None]).astype(np.float32)
    elif case == "mhc.sinkhorn_normalize_fwd":
        rows = int(shape["num_tokens"])
        mhc = int(shape["mhc"])
        repeat = int(shape["repeat"])
        eps = float(shape["eps"])
        x = rng.normal(0.0, 1.0, size=(rows, mhc, mhc)).astype(np.float32)
        generated["x"] = x
        golden["out"] = sinkhorn_forward_np(x, repeat, eps)
    elif case == "mhc.sinkhorn_normalize_bwd":
        rows = int(shape["num_tokens"])
        mhc = int(shape["mhc"])
        repeat = int(shape["repeat"])
        eps = float(shape["eps"])
        grad_output = rng.normal(0.0, 0.25, size=(rows, mhc, mhc)).astype(np.float32)
        x = rng.normal(0.0, 1.0, size=(rows, mhc, mhc)).astype(np.float32)
        generated["grad_output"] = grad_output
        generated["x"] = x
        golden["grad_input"] = sinkhorn_backward_np(grad_output, x, repeat, eps)
    elif case == "mhc.head_compute_mix_fwd":
        rows = int(shape["num_tokens"])
        mhc = int(shape["mhc_mult"])
        eps = np.float32(float(shape["eps"]))
        input_mix = rng.normal(0.0, 1.0, size=(rows, mhc)).astype(np.float32)
        mhc_scale = rng.normal(0.0, 0.5, size=(1,)).astype(np.float32)
        mhc_base = rng.normal(0.0, 0.5, size=(mhc,)).astype(np.float32)
        generated["input_mix"] = input_mix
        generated["mhc_scale"] = mhc_scale
        generated["mhc_base"] = mhc_base
        golden["output_mix"] = sigmoid_np(input_mix * mhc_scale[0] + mhc_base[None, :]) + eps
    elif case == "mhc.head_compute_mix_bwd":
        rows = int(shape["num_tokens"])
        mhc = int(shape["mhc_mult"])
        output_mix_grad = rng.normal(0.0, 0.25, size=(rows, mhc)).astype(np.float32)
        input_mix = rng.normal(0.0, 1.0, size=(rows, mhc)).astype(np.float32)
        mhc_scale = rng.normal(0.0, 0.5, size=(1,)).astype(np.float32)
        mhc_base = rng.normal(0.0, 0.5, size=(mhc,)).astype(np.float32)
        sig = sigmoid_np(input_mix * mhc_scale[0] + mhc_base[None, :])
        grad = output_mix_grad * sig * (np.float32(1.0) - sig)
        generated["output_mix_grad"] = output_mix_grad
        generated["input_mix"] = input_mix
        generated["mhc_scale"] = mhc_scale
        generated["mhc_base"] = mhc_base
        golden["input_mix_grad"] = grad * mhc_scale[0]
        golden["mhc_scale_grad_partial"] = np.array([np.sum(grad * input_mix, dtype=np.float32)], dtype=np.float32)
        golden["mhc_base_grad_partial"] = np.sum(grad, axis=0, dtype=np.float32)
    elif case == "mhc.pre_split_mixes_fwd":
        rows = int(shape["num_tokens"])
        mhc = int(shape["mhc_mult"])
        mhc2 = int(shape["mhc_mult2"])
        mhc3 = int(shape["mhc_mult3"])
        eps = np.float32(float(shape["mhc_pre_eps"]))
        post_mult = np.float32(float(shape["mhc_post_mult_value"]))
        input_mixes = rng.normal(0.0, 1.0, size=(rows, mhc3)).astype(np.float32)
        mhc_scale = rng.normal(0.0, 0.5, size=(3,)).astype(np.float32)
        mhc_base = rng.normal(0.0, 0.5, size=(mhc3,)).astype(np.float32)
        generated["input_mixes"] = input_mixes
        generated["mhc_scale"] = mhc_scale
        generated["mhc_base"] = mhc_base
        golden["pre_layer_mix"] = (
            sigmoid_np(input_mixes[:, :mhc] * mhc_scale[0] + mhc_base[None, :mhc])
            + eps
        )
        golden["post_layer_mix"] = (
            sigmoid_np(input_mixes[:, mhc : 2 * mhc] * mhc_scale[1] + mhc_base[None, mhc : 2 * mhc])
            * post_mult
        )
        golden["comb_res_mix"] = (
            input_mixes[:, 2 * mhc : 2 * mhc + mhc2] * mhc_scale[2]
            + mhc_base[None, 2 * mhc : 2 * mhc + mhc2]
        )
    elif case == "mhc.pre_split_mixes_bwd":
        rows = int(shape["num_tokens"])
        mhc = int(shape["mhc_mult"])
        mhc2 = int(shape["mhc_mult2"])
        mhc3 = int(shape["mhc_mult3"])
        post_mult = np.float32(float(shape["mhc_post_mult_value"]))
        pre_layer_mix_grad = rng.normal(0.0, 0.25, size=(rows, mhc)).astype(np.float32)
        post_layer_mix_grad = rng.normal(0.0, 0.25, size=(rows, mhc)).astype(np.float32)
        comb_res_mix_grad = rng.normal(0.0, 0.25, size=(rows, mhc2)).astype(np.float32)
        input_mixes = rng.normal(0.0, 1.0, size=(rows, mhc3)).astype(np.float32)
        mhc_scale = rng.normal(0.0, 0.5, size=(3,)).astype(np.float32)
        mhc_base = rng.normal(0.0, 0.5, size=(mhc3,)).astype(np.float32)
        post_layer_mix = (
            sigmoid_np(input_mixes[:, mhc : 2 * mhc] * mhc_scale[1] + mhc_base[None, mhc : 2 * mhc])
            * post_mult
        )
        pre_sig = sigmoid_np(input_mixes[:, :mhc] * mhc_scale[0] + mhc_base[None, :mhc])
        pre_unscaled = pre_layer_mix_grad * pre_sig * (np.float32(1.0) - pre_sig)
        post_unscaled = (
            post_layer_mix_grad
            * post_layer_mix
            * (np.float32(1.0) - post_layer_mix / post_mult)
        )
        comb_unscaled = comb_res_mix_grad
        input_mixes_grad = np.empty((rows, mhc3), dtype=np.float32)
        input_mixes_grad[:, :mhc] = pre_unscaled * mhc_scale[0]
        input_mixes_grad[:, mhc : 2 * mhc] = post_unscaled * mhc_scale[1]
        input_mixes_grad[:, 2 * mhc : 2 * mhc + mhc2] = comb_unscaled * mhc_scale[2]
        scale_grad = np.array(
            [
                np.sum(pre_unscaled * input_mixes[:, :mhc], dtype=np.float32),
                np.sum(post_unscaled * input_mixes[:, mhc : 2 * mhc], dtype=np.float32),
                np.sum(comb_unscaled * input_mixes[:, 2 * mhc : 2 * mhc + mhc2], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        base_grad = np.zeros((mhc3,), dtype=np.float32)
        base_grad[:mhc] = np.sum(pre_unscaled, axis=0, dtype=np.float32)
        base_grad[mhc : 2 * mhc] = np.sum(post_unscaled, axis=0, dtype=np.float32)
        base_grad[2 * mhc : 2 * mhc + mhc2] = np.sum(comb_unscaled, axis=0, dtype=np.float32)
        generated["pre_layer_mix_grad"] = pre_layer_mix_grad
        generated["post_layer_mix_grad"] = post_layer_mix_grad
        generated["comb_res_mix_grad"] = comb_res_mix_grad
        generated["input_mixes"] = input_mixes
        generated["post_layer_mix"] = post_layer_mix
        generated["mhc_scale"] = mhc_scale
        generated["mhc_base"] = mhc_base
        golden["input_mixes_grad"] = input_mixes_grad
        golden["mhc_scale_grad_partial"] = scale_grad
        golden["mhc_base_grad_partial"] = base_grad
    else:
        raise ValueError(f"unsupported case {{case}}")

    for name, info in spec["buffers"].items():
        dtype = info["dtype"]
        if info["role"] == "input":
            write_array(output_dir / files[name], dtype, generated[name])
        elif info["role"] == "output":
            write_array(output_dir / files[name], dtype, zeros(dtype, int(info["elements"])))
            write_array(output_dir / golden_files[name], dtype, golden[name])
        elif info["role"] == "inout":
            write_array(output_dir / files[name], dtype, generated[name])
            write_array(output_dir / golden_files[name], dtype, golden[name])
        else:
            raise ValueError(f"unsupported buffer role {{info['role']}} for {{name}}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
"""


def _write_generated_compare(case, spec: dict[str, Any]) -> str:
    files = _arg_files(case)
    golden_files = _golden_files(case)
    payload = json.dumps({"spec": spec, "files": files, "golden_files": golden_files}, indent=2, sort_keys=True)
    return f"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


CASE = json.loads({payload!r})


def bf16_bits_to_f32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def read_array(path: Path, dtype: str):
    if dtype == "f32":
        return np.fromfile(path, dtype=np.float32)
    if dtype == "bf16":
        return bf16_bits_to_f32(np.fromfile(path, dtype=np.uint16))
    if dtype == "i32":
        return np.fromfile(path, dtype=np.int32)
    if dtype == "i64":
        return np.fromfile(path, dtype=np.int64)
    if dtype == "u32":
        return np.fromfile(path, dtype=np.uint32)
    raise ValueError(f"unsupported dtype {{dtype}}")


def compare_one(name: str, info: dict, output_dir: Path) -> bool:
    files = CASE["files"]
    golden_files = CASE["golden_files"]
    dtype = info["dtype"]
    eps = float(info.get("eps", 1e-4))
    output_path = output_dir / files[name]
    golden_path = output_dir / golden_files[name]
    if not output_path.exists():
        print(f"[ERROR] Output missing: {{output_path}}")
        return False
    if not golden_path.exists():
        print(f"[ERROR] Golden missing: {{golden_path}}")
        return False
    output = read_array(output_path, dtype)
    golden = read_array(golden_path, dtype)
    if output.shape != golden.shape:
        print(f"[ERROR] {{name}} shape mismatch: {{output.shape}} vs {{golden.shape}}")
        return False
    if not np.allclose(output, golden, atol=eps, rtol=eps, equal_nan=True):
        diff = np.abs(output.astype(np.float64) - golden.astype(np.float64))
        idx = int(np.argmax(diff))
        print(
            f"[ERROR] {{name}} mismatch: max diff={{float(diff[idx])}} at idx={{idx}} "
            f"(golden={{float(golden[idx])}}, output={{float(output[idx])}}, eps={{eps}})"
        )
        return False
    return True


def main() -> None:
    spec = CASE["spec"]
    output_dir = Path(".")
    ok = True
    for name, info in spec["buffers"].items():
        if info["role"] in {"output", "inout"}:
            ok = compare_one(name, info, output_dir) and ok
    if not ok:
        print("[ERROR] compare failed")
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
"""


def _write_validation_assets(case, config: dict[str, Any], pto_path: Path, ir_text: str, validation_dir: Path) -> str:
    symbol = _kernel_symbol(ir_text)
    specs = _validation_specs(case, config)
    generated_dirs: list[str] = []

    for spec in specs:
        case_dir = _case_validation_dir(validation_dir, case.name, spec)
        generated_dirs.append(str(case_dir))
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "kernel.pto").write_text(ir_text if ir_text.endswith("\n") else f"{ir_text}\n", encoding="utf-8")

        gm_params = ", ".join(_gm_decl(arg, config) for arg in case.args)
        ptr_params = ", ".join(_ptr_decl(arg, config) for arg in case.args)
        launch_args = ", ".join(_kernel_call_arg(arg, config) for arg in case.args)
        void_args = "\n".join(f"  (void){arg.name};" for arg in case.args)
        wrapper = f"Launch_{symbol}"

        stub = f"""#include <stdint.h>
#ifndef __global__
#define __global__
#endif
#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void {symbol}({gm_params}) {{
{void_args}
}}
"""
        launch = f"""#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#include <stdint.h>
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void {symbol}({gm_params});

void {wrapper}({ptr_params}, void *stream) {{
  {symbol}<<<{case.block_dim}, nullptr, stream>>>({launch_args});
}}
"""
        main = _write_generated_main(case, config, symbol, spec)
        golden = _write_generated_golden(case, spec)
        compare = _write_generated_compare(case, spec)

        (case_dir / "stub.cpp").write_text(stub, encoding="utf-8")
        (case_dir / "launch.cpp").write_text(launch, encoding="utf-8")
        (case_dir / "main.cpp").write_text(main, encoding="utf-8")
        (case_dir / "golden.py").write_text(golden, encoding="utf-8")
        (case_dir / "compare.py").write_text(compare, encoding="utf-8")

    return f"{len(generated_dirs)} validation asset(s) under {validation_dir / case.name.replace('.', '/')}"


def build_case(
    case,
    config: dict[str, Any],
    out_dir: Path,
    *,
    ptoas: str | None,
    skip_ptoas: bool,
    ptoas_flags: list[str],
    materialize_validation: bool,
    validation_dir: Path,
) -> BuildResult:
    case_dir = _case_out_dir(out_dir, case.name, config)
    case_dir.mkdir(parents=True, exist_ok=True)

    ir_module = _build_ir(case, config)
    ir_text = f"{ir_module}\n"
    pto_path = case_dir / "kernel.pto"
    cpp_path = case_dir / "kernel.cpp"
    pto_path.write_text(ir_text, encoding="utf-8")

    validation_note = None
    if materialize_validation:
        validation_case_dir = _write_validation_assets(case, config, pto_path, ir_text, validation_dir)
        validation_note = f"validation assets: {validation_case_dir}"

    if skip_ptoas:
        return BuildResult(
            case=case.name,
            config=config,
            pto_path=str(pto_path),
            cpp_path=None,
            ptoas_status="skipped",
            ptoas_command=None,
            note=validation_note or "Generated PTO only; ptoas was skipped.",
        )

    ptoas_bin = _resolve_ptoas(ptoas)
    cmd = _run_ptoas(ptoas_bin, pto_path, cpp_path, _ptoas_flags(ptoas_flags))
    return BuildResult(
        case=case.name,
        config=config,
        pto_path=str(pto_path),
        cpp_path=str(cpp_path),
        ptoas_status="passed",
        ptoas_command=cmd,
        note=validation_note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TileKernels PTO-DSL migration cases.")
    parser.add_argument("--case", action="append", dest="cases", help="Case name to build, for example moe.normalize_weight. Can be repeated.")
    parser.add_argument("--list", action="store_true", help="List registered cases and exit.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ptoas", help="Path to ptoas. Defaults to PTOAS_BIN or PATH.")
    parser.add_argument("--skip-ptoas", action="store_true", help="Only generate .pto files.")
    parser.add_argument("--ptoas-flag", action="append", default=[], help="Extra flag passed before input .pto. Can be repeated.")
    parser.add_argument("--materialize-validation", action="store_true", help="Write PTO-Gym validation case assets for each built config.")
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR, help="Output root for generated validation assets.")
    args = parser.parse_args()

    registry = _load_registry()
    selected_names = set(args.cases or [])
    selected = [case for case in registry if not selected_names or case.name in selected_names]

    if args.list:
        for case in registry:
            print(f"{case.name}\t{case.status}\t{case.description}")
        return

    missing = selected_names.difference({case.name for case in registry})
    if missing:
        raise SystemExit(f"Unknown case(s): {', '.join(sorted(missing))}")

    results: list[BuildResult] = []
    for case in selected:
        if case.status != "implemented":
            continue
        for config in case.configs:
            results.append(
                build_case(
                    case,
                    dict(config),
                    args.out_dir,
                    ptoas=args.ptoas,
                    skip_ptoas=args.skip_ptoas,
                    ptoas_flags=list(args.ptoas_flag),
                    materialize_validation=args.materialize_validation,
                    validation_dir=args.validation_dir,
                )
            )

    report_path = args.out_dir / "build-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for result in results:
        print(f"{result.case} {result.config} -> {result.ptoas_status} ({result.pto_path})")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
