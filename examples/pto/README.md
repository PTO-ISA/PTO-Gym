# VPTO Validation

`examples/pto` is the entry point for PTO validation on SIM or NPU.

## Required Environment

The runner depends on `ASCEND_HOME_PATH` and `PTOAS_BIN`.

Set `DEVICE=SIM` for simulator runs or `DEVICE=NPU` for hardware runs.

`SIM_LIB_DIR` is an optional environment variable for `DEVICE=SIM`. When it
is unset, the runner auto-discovers `*/simulator/dav_3510/lib` under
`ASCEND_HOME_PATH`.

## Run One Case

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

Use `DEVICE=NPU` to run the same case on hardware.

## Run Micro-Op Validation

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

Use `DEVICE=NPU` to run the same batch on hardware.

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
