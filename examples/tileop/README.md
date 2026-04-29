# TileOp Validation

`examples/tileop` packages the A5 TileLang ST TileOp examples together with
standalone runners for SIM and NPU validation.

## Required Environment

The runners depend on:

- `ASCEND_HOME_PATH`: your local CANN installation root
- `PTOAS_BIN`: path to the `ptoas` executable, or pass it with `--ptoas-bin`

The scripts derive `bisheng`, `cce-ld`, `ld.lld`, and simulator libraries from `ASCEND_HOME_PATH`. If `${ASCEND_HOME_PATH}/bin/setenv.bash` exists, the runners source it automatically.

The commands below assume your current directory is `examples/tileop/`.

## Run One Testcase

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export PTOAS_BIN=/path/to/ptoas

python3 script/run_example.py -r sim -v a5 -t tadd
```

Run one concrete case inside a testcase:

```bash
python3 script/run_example.py \
  -r sim \
  -v a5 \
  -t tadd \
  -c f32_16x64
```

Use `-r npu` to run on hardware instead of simulator.

## Find Runnable Testcases

Runnable testcases are the example directories under `src/testcase/`. Use the directory name as the `-t/--testcase` value for `script/run_example.py`.

For example, `src/testcase/tadd/` maps to:

```bash
python3 script/run_example.py -r sim -v a5 -t tadd
```

To see the available testcase names from `examples/tileop/`, run:

```bash
find src/testcase -mindepth 1 -maxdepth 1 -type d | xargs -n1 basename | sort
```

## Workspace

Build products, generated input/output data, and comparison artifacts are written to a separate workspace instead of `examples/tileop/`.

Override it with either:

```bash
export WORK_SPACE=/tmp/pto-gym-tileop-work
```

or:

```bash
python3 script/run_example.py \
  -r sim \
  -v a5 \
  -t tadd \
  --work-dir /tmp/pto-gym-tileop-work
```

If no workspace is provided, the runners default to `/tmp/pto-gym-tileop/...`.

## Notes

- `sim` mode requires simulator libraries under `${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib`.
- `--without-build` reuses the existing workspace build directory.
