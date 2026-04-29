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
- [docs/PTO-tile-Instruction-SPEC.md](docs/PTO-tile-Instruction-SPEC.md)

## Tests as Learning Material

The PTO micro Instruction test cases under [examples/pto/](examples/pto/) are both validation assets and learning material for PTO developers.

Each runnable case follows a stable structure:

- `kernel.pto`: PTO kernel source, defining the device-side computation logic.
- `stub.cpp`: host-side stub used to embed and expose the generated device object/symbols.
- `launch.cpp`: launch wrapper that prepares launch parameters and invokes the kernel entry.
- `main.cpp`: executable entry on host side, orchestrating data preparation, launch, and end-to-end flow.
- `golden.py`: generates test inputs and expected (golden) outputs for validation.
- `compare.py`: compares runtime outputs against golden results and reports pass/fail.

These cases currently focus on PTO micro-op scenarios and are useful for understanding instruction behavior through concrete examples.

For more detailed validation guidance, see [examples/pto/README.md](examples/pto/README.md).

Tile Instruction ST cases are provided under [examples/tileop/](examples/tileop/). They package the TileLang ST A5 examples together with runners that:

- take `PTOAS_BIN` from `--ptoas-bin` or the environment
- discover `bisheng` and related CANN tools from `ASCEND_HOME_PATH`
- keep build and generated data under a separate workspace instead of writing into the source tree

For usage details, see [examples/tileop/README.md](examples/tileop/README.md).

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

### Run one Tile Instruction testcase

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export PTOAS_BIN=/path/to/ptoas

python3 examples/tileop/script/run_example.py -r sim -v a5 -t tadd
```

## Repository Layout

- [docs/](docs/) — PTO instruction SPEC documentation
- [examples/pto/README.md](examples/pto/README.md) — VPTO validation usage guide
- [examples/pto/](examples/pto/) — VPTO learning and validation cases
- [examples/tileop/README.md](examples/tileop/README.md) — Tile Instruction ST usage guide
- [examples/tileop/](examples/tileop/) — Tile Instruction learning and validation cases

## License

See [LICENSE](LICENSE) for license terms.
