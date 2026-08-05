# VPTO Validation

`examples/pto` is the entry point for PTO validation on SIM or NPU.

The checked-in suite is exported from PTOAS commit `d5072e79`. It contains the
five-file `kernel.pto` runtime cases from `test/vpto/cases`, plus the Flash
Attention learning case adapted from `test/samples/FlashAttention`. PTODSL-only
`kernel.py` cases remain in PTOAS because this standalone runner does not ship
the PTODSL simulator environment.

## Required Environment

The runner depends on `ASCEND_HOME_PATH` and `PTOAS_BIN`.

Use CANN 9.0.0 official or newer. `pto.ld_dev` and `pto.st_dev` intentionally
reject beta releases. Some `onboard-only/` SIMT state-preservation cases need
CANN 9.1.0 official for device compilation.

PTOAS is invoked through the VPTO backend by default. Set the same contract
explicitly when invoking the runner from automation:

```bash
export PTOAS_FLAGS='--pto-backend=vpto --pto-arch a5'
```

When a case provides `ptoas.flags`, that file is authoritative. This is used
for cases that select backends per nested module.

Set `DEVICE=SIM` for simulator runs or `DEVICE=NPU` for hardware runs.

`SIM_LIB_DIR` is an optional environment variable for `DEVICE=SIM`. When it
is unset, the runner auto-discovers `*/simulator/dav_3510/lib` under
`ASCEND_HOME_PATH`.

## Run One Case

```bash
mkdir -p .work/vpto-single

WORK_SPACE=$PWD/.work/vpto-single \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
PTOAS_FLAGS='--pto-backend=vpto --pto-arch a5' \
CASE_NAME=micro-op/binary-vector/vadd \
DEVICE=SIM \
bash examples/pto/scripts/run_host_vpto_validation.sh
```

Use `DEVICE=NPU` to run the same case on hardware.

## Run Micro-Op Validation

```bash
mkdir -p .work/vpto-sim-microop-64

WORK_SPACE=$PWD/.work/vpto-sim-microop-64 \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
PTOAS_FLAGS='--pto-backend=vpto --pto-arch a5' \
CASE_PREFIX=micro-op \
DEVICE=SIM \
JOBS=64 \
bash examples/pto/scripts/run_host_vpto_validation_parallel.sh
```

Use `DEVICE=NPU` to run the same batch on hardware.

Cases under `onboard-only/` are excluded from SIM runtime sweeps. They remain
available for NPU runs and `COMPILE_ONLY=1` validation.

## Results

Single-case logs are written under `WORK_SPACE/<case-token>/validation.log`.

Parallel runs write:

```text
$WORK_SPACE/parallel-runner.log
$WORK_SPACE/parallel-summary.tsv
```

## Useful Overrides

```bash
export CASE_NAME=micro-op/binary-vector/vadd
export CASE_PREFIX=micro-op
export DEVICE=SIM
export JOBS=64
export SIM_LIB_DIR=/path/to/simulator/dav_3510/lib  # optional for DEVICE=SIM
```
