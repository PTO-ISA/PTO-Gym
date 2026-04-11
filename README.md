# PTO-Gym

Tile operation tutorials and examples for PTO-based programming.

## Overview

PTO-Gym is a developer-facing repository for PTO tile programming resources. It currently provides three core capabilities:

- `ptoas` binary / wheel release entry for users who want ready-to-use assembler artifacts
- PTO instruction SPEC documentation for understanding the instruction model and semantics
- PTO test cases under `examples/` that help developers learn PTO tile instructions and micro-ops through runnable examples

## Prerequisites

This repository depends on the CANN package for validation and learning workflows.

- Recommended CANN version: `9.0.0-beta.1`
- Validation scripts use `ASCEND_HOME_PATH` to locate your local CANN installation
- Example:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
```

If your CANN installation provides `set_env.sh`, the validation scripts will source it automatically.

## Binary Releases

You can obtain `ptoas` binaries and related release artifacts from the **Releases / Packages** area on the right side of the GitHub repository page.

## PTO Instruction SPEC

This repository provides PTO instruction SPEC documentation here:

- [docs/PTO-micro-Instruction-SPEC.md](docs/PTO-micro-Instruction-SPEC.md)

## Tests as Learning Material

The PTO micro Instruction test cases under [examples/pto/](examples/pto/) are both validation assets and learning material for PTO developers.

Each runnable case follows a stable structure:

- `kernel.pto`
- `stub.cpp`
- `launch.cpp`
- `main.cpp`
- `golden.py`
- `compare.py`

These cases currently focus on PTO micro-op scenarios and are useful for understanding instruction behavior through concrete examples. Tile-level cases may be added in future updates.

For more detailed validation guidance, see [examples/pto/README.md](examples/pto/README.md).

## Quick Start for Validation

### Run one case

```bash
mkdir -p .work/vpto-single
rm -rf .work/vpto-single/*

WORK_SPACE=$PWD/.work/vpto-single \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
CASE_NAME=micro-op/binary-vector/vadd \
DEVICE=SIM \
bash examples/pto/scripts/run_host_vpto_validation.sh
```

### Run micro-op validation in parallel

```bash
mkdir -p .work/vpto-sim-microop-64
rm -rf .work/vpto-sim-microop-64/*

WORK_SPACE=$PWD/.work/vpto-sim-microop-64 \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
CASE_PREFIX=micro-op \
DEVICE=SIM \
JOBS=64 \
bash examples/pto/scripts/run_host_vpto_validation_parallel.sh
```

## Repository Layout

- [docs/](docs/) — PTO instruction SPEC documentation
- [examples/pto/README.md](examples/pto/README.md) — VPTO validation usage guide
- [examples/pto/](examples/pto/) — VPTO learning and validation cases
