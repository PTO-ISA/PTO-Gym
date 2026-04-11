# VPTO Validation

`test/vpto` is the entry point for VPTO simulator validation.

## Required Environment

The runner depends on `ASCEND_HOME_PATH` and `PTOAS_BIN`.

When `DEVICE=SIM` and `SIM_LIB_DIR` is unset, the runner auto-discovers
`*/simulator/dav_3510/lib` under `ASCEND_HOME_PATH`.

## Run One Case

```bash
mkdir -p .work/vpto-single
rm -rf .work/vpto-single/*

WORK_SPACE=$PWD/.work/vpto-single \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
CASE_NAME=micro-op/binary-vector/vadd \
DEVICE=SIM \
bash test/vpto/scripts/run_host_vpto_validation.sh
```

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
bash test/vpto/scripts/run_host_vpto_validation_parallel.sh
```

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
export SIM_LIB_DIR=/path/to/simulator/dav_3510/lib
```
