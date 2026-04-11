#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VPTO_ROOT="${ROOT_DIR}/examples/pto"
CASES_ROOT="${VPTO_ROOT}"
SERIAL_SCRIPT="${SCRIPT_DIR}/run_host_vpto_validation.sh"

WORK_SPACE="${WORK_SPACE:-}"
CASE_NAME="${CASE_NAME:-}"
CASE_PREFIX="${CASE_PREFIX:-}"
JOBS="${JOBS:-}"

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${SERIAL_SCRIPT}" ]] || die "missing serial validation script: ${SERIAL_SCRIPT}"
[[ -d "${CASES_ROOT}" ]] || die "missing cases root: ${CASES_ROOT}"
[[ -n "${WORK_SPACE}" ]] || die "WORK_SPACE is required"

if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS=1
  fi
  if [[ "${JOBS}" -gt 1 ]]; then
    JOBS="$((JOBS / 2))"
  fi
fi

[[ "${JOBS}" =~ ^[0-9]+$ ]] || die "JOBS must be a positive integer, got: ${JOBS}"
[[ "${JOBS}" -ge 1 ]] || die "JOBS must be >= 1"

mkdir -p "${WORK_SPACE}"
WORK_SPACE="$(cd "${WORK_SPACE}" && pwd)"
SUMMARY_FILE="${WORK_SPACE}/parallel-summary.tsv"
RUNNER_LOG="${WORK_SPACE}/parallel-runner.log"

discover_cases() {
  local required_files=(
    kernel.pto
    stub.cpp
    launch.cpp
    main.cpp
    golden.py
    compare.py
  )

  if [[ -n "${CASE_NAME}" ]]; then
    local requested_dir="${CASES_ROOT}/${CASE_NAME}"
    [[ -d "${requested_dir}" ]] || die "unknown case: ${CASE_NAME}"
    for f in "${required_files[@]}"; do
      [[ -f "${requested_dir}/${f}" ]] || die "case ${CASE_NAME} is missing ${f}"
    done
    printf "%s\n" "${CASE_NAME#/}"
    return 0
  fi

  find "${CASES_ROOT}" -mindepth 1 -type d | sort | while read -r dir; do
    local ok=1
    for f in "${required_files[@]}"; do
      if [[ ! -f "${dir}/${f}" ]]; then
        ok=0
        break
      fi
    done
    [[ "${ok}" -eq 1 ]] || continue
    local rel="${dir#${CASES_ROOT}/}"
    if [[ -n "${CASE_PREFIX}" && "${rel}" != "${CASE_PREFIX}"* ]]; then
      continue
    fi
    printf "%s\n" "${rel}"
  done
}

readarray -t CASES < <(discover_cases)
[[ "${#CASES[@]}" -gt 0 ]] || die "no cases found under ${CASES_ROOT}"

: > "${SUMMARY_FILE}"
: > "${RUNNER_LOG}"

declare -A PID_TO_CASE=()

launch_case() {
  local case_name="$1"

  log "[${case_name}] launch" | tee -a "${RUNNER_LOG}"
  (
    CASE_NAME="${case_name}" "${SERIAL_SCRIPT}"
  ) &

  local pid=$!
  PID_TO_CASE["${pid}"]="${case_name}"
}

reap_one() {
  local pid="$1"
  local case_name="${PID_TO_CASE[${pid}]}"
  local result="FAIL"
  local detail="1"

  if wait "${pid}"; then
    result="PASS"
    detail="0"
  fi

  printf '%s\t%s\t%s\n' "${case_name}" "${result}" "${detail}" >> "${SUMMARY_FILE}"
  log "[${case_name}] ${result} (${detail})" | tee -a "${RUNNER_LOG}"
  unset 'PID_TO_CASE['"${pid}"']'
}

log "=== VPTO Host Validation Parallel ===" | tee -a "${RUNNER_LOG}"
log "WORK_SPACE=${WORK_SPACE}" | tee -a "${RUNNER_LOG}"
log "CASE_NAME=${CASE_NAME:-<all>}" | tee -a "${RUNNER_LOG}"
log "CASE_PREFIX=${CASE_PREFIX:-<none>}" | tee -a "${RUNNER_LOG}"
log "JOBS=${JOBS}" | tee -a "${RUNNER_LOG}"
log "TOTAL_CASES=${#CASES[@]}" | tee -a "${RUNNER_LOG}"

next_index=0
while [[ "${next_index}" -lt "${#CASES[@]}" || "${#PID_TO_CASE[@]}" -gt 0 ]]; do
  while [[ "${next_index}" -lt "${#CASES[@]}" && "${#PID_TO_CASE[@]}" -lt "${JOBS}" ]]; do
    launch_case "${CASES[${next_index}]}"
    next_index="$((next_index + 1))"
  done

  if [[ "${#PID_TO_CASE[@]}" -eq 0 ]]; then
    continue
  fi

  while true; do
    for pid in "${!PID_TO_CASE[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        reap_one "${pid}"
        break 2
      fi
    done
    sleep 1
  done
done

pass_count="$(awk -F '\t' '$2 == "PASS" {count++} END {print count + 0}' "${SUMMARY_FILE}")"
fail_count="$(awk -F '\t' '$2 != "PASS" {count++} END {print count + 0}' "${SUMMARY_FILE}")"

log "PASS=${pass_count} FAIL=${fail_count}" | tee -a "${RUNNER_LOG}"
log "summary: ${SUMMARY_FILE}" | tee -a "${RUNNER_LOG}"

if [[ "${fail_count}" -ne 0 ]]; then
  die "parallel validation finished with ${fail_count} failing case(s)"
fi

log "All ${pass_count} case(s) passed" | tee -a "${RUNNER_LOG}"
