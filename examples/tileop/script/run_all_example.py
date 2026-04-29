#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Batch runner for TileOp examples in PTO-Gym."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import tempfile
import traceback

import run_example


SOC_VERSION_MAP = {
    "a5": "Ascend950PR_9599",
}


def discover_testcases(testcase_root):
    testcases = []
    for entry in sorted(os.listdir(testcase_root)):
        testcase_dir = os.path.join(testcase_root, entry)
        if not os.path.isdir(testcase_dir):
            continue
        pto_file = os.path.join(testcase_dir, f"{entry}.pto")
        if os.path.isfile(pto_file):
            testcases.append(entry)
    return testcases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all TileOp examples for CI or local batch validation."
    )
    parser.add_argument(
        "-r", "--run-mode", default="sim", help="Run mode: sim or npu (default: sim)"
    )
    parser.add_argument(
        "-v", "--soc-version", default="a5", help="SoC version: a5 (default: a5)"
    )
    parser.add_argument("-p", "--ptoas-bin", default=None, help="Path to ptoas binary")
    parser.add_argument(
        "-t",
        "--testcase",
        action="append",
        default=[],
        help="Run only selected testcase(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "-w",
        "--without-build",
        action="store_true",
        help="Skip build and reuse the existing build directory.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Workspace for build and generated data; defaults to /tmp/pto-gym-tileop/...",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop immediately after the first failed testcase."
    )
    parser.add_argument("--list", action="store_true", help="List discovered testcases and exit.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of testcases to run in parallel after the shared build (default: 1).",
    )
    return parser.parse_args()


def resolve_selected_testcases(all_testcases, requested):
    if not requested:
        return all_testcases

    requested_set = []
    seen = set()
    for testcase in requested:
        if testcase not in seen:
            requested_set.append(testcase)
            seen.add(testcase)

    missing = [testcase for testcase in requested_set if testcase not in all_testcases]
    if missing:
        raise ValueError(
            f"Unsupported testcase(s): {', '.join(missing)}; supported: {', '.join(all_testcases)}"
        )
    return requested_set


def resolve_work_root(args):
    explicit = args.work_dir or os.environ.get("WORK_SPACE")
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(
        tempfile.gettempdir(),
        "pto-gym-tileop",
        args.soc_version,
        args.run_mode,
        "batch",
    )


def run_testcase_subprocess(script_path, run_mode, soc_version, ptoas_bin, testcase, work_root):
    command = [
        sys.executable,
        script_path,
        "-r",
        run_mode,
        "-v",
        soc_version,
        "-t",
        testcase,
        "-p",
        ptoas_bin,
        "--work-dir",
        work_root,
        "-w",
    ]
    env = os.environ.copy()
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return testcase, result.returncode, result.stdout


def main():
    args = parse_args()

    if args.soc_version not in SOC_VERSION_MAP:
        print(
            f"[ERROR] Unsupported soc-version: {args.soc_version}, supported: {', '.join(sorted(SOC_VERSION_MAP))}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.jobs < 1:
        print("[ERROR] --jobs must be >= 1", file=sys.stderr)
        sys.exit(1)

    script_path = os.path.abspath(__file__)
    tileop_root = os.path.dirname(os.path.dirname(script_path))
    source_dir = run_example.resolve_source_dir(tileop_root, args.soc_version)
    testcase_root = os.path.join(source_dir, "testcase")

    if not os.path.isdir(testcase_root):
        print(f"[ERROR] Testcase root not found: {testcase_root}", file=sys.stderr)
        sys.exit(1)

    all_testcases = discover_testcases(testcase_root)
    if not all_testcases:
        print(f"[ERROR] No testcases found in: {testcase_root}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        for testcase in all_testcases:
            print(testcase)
        return

    try:
        selected_testcases = resolve_selected_testcases(all_testcases, args.testcase)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    ptoas_bin = args.ptoas_bin or run_example.find_ptoas_bin()
    if not ptoas_bin:
        print(
            "[ERROR] Cannot find ptoas binary. Set PTOAS_BIN or use -p/--ptoas-bin.",
            file=sys.stderr,
        )
        sys.exit(1)
    ptoas_bin = run_example.require_executable(ptoas_bin, "ptoas")

    work_root = resolve_work_root(args)
    os.makedirs(work_root, exist_ok=True)

    default_soc_version = SOC_VERSION_MAP[args.soc_version]
    print(f"[INFO] run_mode={args.run_mode}")
    print(f"[INFO] soc_version={args.soc_version} ({default_soc_version})")
    print(f"[INFO] ptoas={ptoas_bin}")
    print(f"[INFO] source_dir={source_dir}")
    print(f"[INFO] work_root={work_root}")
    print(f"[INFO] selected_testcases={', '.join(selected_testcases)}")
    print(f"[INFO] jobs={args.jobs}")

    failures = []
    try:
        bisheng_bin = run_example.set_env_variables(args.run_mode, default_soc_version)
        print(f"[INFO] bisheng={bisheng_bin}")

        if not args.without_build:
            if len(selected_testcases) == 1:
                build_target = selected_testcases[0]
            else:
                build_target = "all"
            print(f"[INFO] build requested for {build_target}")
            run_example.build_project(
                source_dir,
                work_root,
                args.run_mode,
                default_soc_version,
                build_target,
                ptoas_bin,
                bisheng_bin,
            )

        total = len(selected_testcases)
        if args.jobs == 1:
            for index, testcase in enumerate(selected_testcases, start=1):
                print(f"[INFO] [{index}/{total}] running testcase: {testcase}")
                try:
                    run_example.run_gen_data(work_root, testcase_root, testcase)
                    run_example.run_binary(work_root, testcase)
                    run_example.run_compare(work_root, testcase)
                except Exception as exc:
                    failures.append((testcase, str(exc)))
                    print(f"[ERROR] testcase failed: {testcase}")
                    traceback.print_exc()
                    if args.fail_fast:
                        break
        else:
            print(f"[INFO] running testcases in parallel with jobs={args.jobs}")
            max_workers = min(args.jobs, total)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_testcase = {}
                for index, testcase in enumerate(selected_testcases, start=1):
                    print(f"[INFO] [{index}/{total}] queue testcase: {testcase}")
                    future = executor.submit(
                        run_testcase_subprocess,
                        script_path,
                        args.run_mode,
                        args.soc_version,
                        ptoas_bin,
                        testcase,
                        work_root,
                    )
                    future_to_testcase[future] = testcase

                for future in concurrent.futures.as_completed(future_to_testcase):
                    testcase = future_to_testcase[future]
                    try:
                        _, returncode, output = future.result()
                    except Exception as exc:
                        failures.append((testcase, str(exc)))
                        print(f"[ERROR] testcase runner crashed: {testcase}")
                        traceback.print_exc()
                        if args.fail_fast:
                            break
                        continue

                    print(f"[INFO] ===== testcase {testcase} output begin =====")
                    if output:
                        print(output, end="" if output.endswith("\n") else "\n")
                    print(f"[INFO] ===== testcase {testcase} output end =====")

                    if returncode != 0:
                        failures.append((testcase, f"subprocess exited with {returncode}"))
                        print(f"[ERROR] testcase failed: {testcase}")
                        if args.fail_fast:
                            break

    except Exception as exc:
        print(f"[ERROR] batch run failed: {exc}", file=sys.stderr)
        sys.exit(1)

    passed = len(selected_testcases) - len(failures)
    print("[INFO] TileOp ST summary")
    print(f"[INFO] passed={passed} failed={len(failures)} total={len(selected_testcases)}")
    if failures:
        for testcase, reason in failures:
            print(f"[INFO] failed testcase: {testcase} ({reason})")
        sys.exit(1)


if __name__ == "__main__":
    main()
