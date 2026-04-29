#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Runner for a single TileOp example in PTO-Gym."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile


SOC_VERSION_MAP = {
    "a5": "Ascend950PR_9599",
}


def run_command(command, cwd=None, check=True):
    try:
        print(f"run command: {' '.join(command)}")
        subprocess.run(command, cwd=cwd, check=check, stdout=None, stderr=None, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"run command failed with return code {exc.returncode}")
        raise


def require_executable(path, description):
    if not path:
        raise FileNotFoundError(f"{description} is not set")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{description} is not executable: {path}")
    return os.path.abspath(path)


def find_ptoas_bin():
    env_bin = os.environ.get("PTOAS_BIN")
    if env_bin and os.path.isfile(env_bin):
        return os.path.abspath(env_bin)
    return None


def prepend_env_path(var_name, path):
    if not path:
        return
    normalized = os.path.realpath(path)
    entries = [entry for entry in os.environ.get(var_name, "").split(":") if entry]
    filtered = [entry for entry in entries if os.path.realpath(entry) != normalized]
    os.environ[var_name] = ":".join([normalized] + filtered)


def add_ptoas_lib_dir(ptoas_bin):
    lib_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(ptoas_bin)), "..", "lib"))
    if not os.path.isdir(lib_dir):
        print(f"warning: ptoas lib dir not found: {lib_dir}")
        return
    prepend_env_path("LD_LIBRARY_PATH", lib_dir)


def import_shell_env(script_path):
    quoted = shlex.quote(script_path)
    result = subprocess.run(
        f"source {quoted} >/dev/null 2>&1 && env",
        shell=True,
        executable=shutil.which("bash") or "bash",
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def set_env_variables(run_mode, soc_version, ptoas_bin):
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set")
    ascend_home = os.path.abspath(ascend_home)
    os.environ["ASCEND_HOME_PATH"] = ascend_home

    setenv_path = os.path.join(ascend_home, "bin", "setenv.bash")
    if os.path.exists(setenv_path):
        print(f"run env shell: {setenv_path}")
        import_shell_env(setenv_path)
    else:
        print(f"warning: not found {setenv_path}")

    bisheng_bin = require_executable(
        os.path.join(ascend_home, "bin", "bisheng"),
        "bisheng",
    )
    os.environ["BISHENG_BIN"] = bisheng_bin
    os.environ["BISHENG_CC1_BIN"] = require_executable(
        os.path.join(ascend_home, "tools", "bisheng_compiler", "bin", "bisheng"),
        "bisheng cc1",
    )
    os.environ["CCE_LD_BIN"] = require_executable(
        os.path.join(ascend_home, "bin", "cce-ld"),
        "cce-ld",
    )
    os.environ["LD_LLD_BIN"] = require_executable(
        os.path.join(ascend_home, "bin", "ld.lld"),
        "ld.lld",
    )

    path_entries = os.environ.get("PATH", "").split(":") if os.environ.get("PATH") else []
    bisheng_dir = os.path.dirname(bisheng_bin)
    if bisheng_dir not in path_entries:
        os.environ["PATH"] = ":".join([bisheng_dir] + path_entries)

    if run_mode == "sim":
        simulator_lib_path = os.path.join(
            ascend_home, "tools", "simulator", soc_version, "lib"
        )
        if not os.path.isdir(simulator_lib_path):
            raise EnvironmentError(f"simulator lib dir not found: {simulator_lib_path}")

        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        filtered_paths = [
            path for path in ld_lib_path.split(":")
            if path and "/runtime/lib64" not in path
        ]
        sim_paths = [
            simulator_lib_path,
            os.path.join(ascend_home, "runtime", "lib64", "stub"),
        ]
        os.environ["LD_LIBRARY_PATH"] = ":".join(sim_paths + filtered_paths)

    add_ptoas_lib_dir(ptoas_bin)
    return bisheng_bin


def resolve_work_root(run_mode, soc_version, testcase, explicit_work_dir):
    work_root = explicit_work_dir or os.environ.get("WORK_SPACE")
    if work_root:
        return os.path.abspath(work_root)
    return os.path.join(
        tempfile.gettempdir(),
        "pto-gym-tileop",
        soc_version,
        run_mode,
        testcase,
    )


def resolve_source_dir(tileop_root, soc_version):
    del soc_version
    return os.path.join(tileop_root, "src")


def get_build_dir(work_root):
    return os.path.join(work_root, "build")


def get_testcase_work_dir(work_root, testcase):
    return os.path.join(get_build_dir(work_root), "testcase", testcase)


def build_project(source_dir, work_root, run_mode, soc_version, testcase, ptoas_bin, bisheng_bin):
    build_dir = get_build_dir(work_root)
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        cmake_cmd = [
            "cmake",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={soc_version}",
            f"-DTEST_CASE={testcase}",
            f"-DPTOAS_BIN={ptoas_bin}",
            f"-DBISHENG_BIN={bisheng_bin}",
            source_dir,
        ]
        subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )

        cpu_count = os.cpu_count() or 4
        make_cmd = ["make", "VERBOSE=1", "-j", str(cpu_count)]
        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        print("compile process:\n", result.stdout)
    except subprocess.CalledProcessError as exc:
        print(f"build failed: {exc.stdout}")
        raise


def copy_testcase_scripts(testcase_root, work_root, testcase):
    work_dir = get_testcase_work_dir(work_root, testcase)
    os.makedirs(work_dir, exist_ok=True)

    shared_src = os.path.join(testcase_root, "st_common.py")
    if os.path.isfile(shared_src):
        shutil.copy2(shared_src, os.path.join(work_dir, "st_common.py"))

    testcase_src = os.path.join(testcase_root, testcase)
    for name in ("cases.py", "gen_data.py", "compare.py"):
        src = os.path.join(testcase_src, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(work_dir, name))


def run_gen_data(work_root, testcase_root, testcase):
    original_dir = os.getcwd()
    try:
        work_dir = get_testcase_work_dir(work_root, testcase)
        copy_testcase_scripts(testcase_root, work_root, testcase)
        os.chdir(work_dir)
        run_command([sys.executable, "gen_data.py"])
    except Exception as exc:
        print(f"gen golden failed: {exc}")
        raise
    finally:
        os.chdir(original_dir)


def run_binary(work_root, testcase, case_filter=None):
    original_dir = os.getcwd()
    try:
        os.chdir(get_testcase_work_dir(work_root, testcase))
        cmd = [os.path.join("..", "..", "bin", testcase)]
        if case_filter:
            cmd.append(case_filter)
        run_command(cmd)
    except Exception as exc:
        print(f"run binary failed: {exc}")
        raise
    finally:
        os.chdir(original_dir)


def run_compare(work_root, testcase, case_filter=None):
    original_dir = os.getcwd()
    try:
        work_dir = get_testcase_work_dir(work_root, testcase)
        os.chdir(work_dir)
        cmd = [sys.executable, "compare.py"]
        if case_filter:
            cmd.append(case_filter)
        run_command(cmd)
    except Exception as exc:
        print(f"compare failed: {exc}")
        raise
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description="TileOp ST runner")
    parser.add_argument("-r", "--run-mode", required=True, help="Run mode: sim or npu")
    parser.add_argument("-v", "--soc-version", required=True, help="SoC version: a5")
    parser.add_argument("-t", "--testcase", required=True, help="Test case name, for example tadd")
    parser.add_argument("-p", "--ptoas-bin", required=False, help="Path to ptoas binary")
    parser.add_argument(
        "-c",
        "--case",
        required=False,
        default=None,
        help="Run a specific case within the testcase, for example f32_16x64",
    )
    parser.add_argument(
        "-w",
        "--without-build",
        action="store_true",
        help="Skip build and reuse the existing build directory",
    )
    parser.add_argument(
        "--work-dir",
        required=False,
        help="Workspace for build and generated data; defaults to /tmp/pto-gym-tileop/...",
    )
    args = parser.parse_args()

    default_soc_version = SOC_VERSION_MAP.get(args.soc_version)
    if not default_soc_version:
        print(
            f"[ERROR] Unsupported soc-version: {args.soc_version}, supported: {', '.join(sorted(SOC_VERSION_MAP))}",
            file=sys.stderr,
        )
        sys.exit(1)

    ptoas_bin = args.ptoas_bin or find_ptoas_bin()
    if not ptoas_bin:
        print(
            "[ERROR] Cannot find ptoas binary. Set PTOAS_BIN or use -p/--ptoas-bin.",
            file=sys.stderr,
        )
        sys.exit(1)
    ptoas_bin = require_executable(ptoas_bin, "ptoas")
    print(f"[INFO] ptoas: {ptoas_bin}")

    script_path = os.path.abspath(__file__)
    tileop_root = os.path.dirname(os.path.dirname(script_path))
    source_dir = resolve_source_dir(tileop_root, args.soc_version)
    testcase_root = os.path.join(source_dir, "testcase")
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Target dir not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    work_root = resolve_work_root(args.run_mode, args.soc_version, args.testcase, args.work_dir)
    os.makedirs(work_root, exist_ok=True)
    print(f"[INFO] source_dir: {source_dir}")
    print(f"[INFO] work_root: {work_root}")

    try:
        bisheng_bin = set_env_variables(args.run_mode, default_soc_version, ptoas_bin)
        print(f"[INFO] bisheng: {bisheng_bin}")

        if not args.without_build:
            build_project(
                source_dir,
                work_root,
                args.run_mode,
                default_soc_version,
                args.testcase,
                ptoas_bin,
                bisheng_bin,
            )

        run_gen_data(work_root, testcase_root, args.testcase)
        run_binary(work_root, args.testcase, args.case)
        run_compare(work_root, args.testcase, args.case)
    except Exception as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
