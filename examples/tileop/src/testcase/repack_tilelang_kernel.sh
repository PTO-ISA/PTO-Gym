#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <ascend_home> <aicore_arch> <kernel_stub_src> <device_obj> <output_obj> <module_id>" >&2
  exit 1
fi

ASCEND_HOME_PATH="$1"
AICORE_ARCH="$2"
KERNEL_STUB_SRC="$3"
DEVICE_OBJ="$4"
OUTPUT_OBJ="$5"
MODULE_ID="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../../../../" && pwd)"

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
BISHENG_CC1_BIN="${BISHENG_CC1_BIN:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
CCE_LD_BIN="${CCE_LD_BIN:-${ASCEND_HOME_PATH}/bin/cce-ld}"
LD_LLD_BIN="${LD_LLD_BIN:-${ASCEND_HOME_PATH}/bin/ld.lld}"
CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/lib/clang/15.0.5}"
CCE_STUB_DIR="${CCE_STUB_DIR:-${CLANG_RESOURCE_DIR}/include/cce_stub}"

HOST_ARCH="$(uname -m)"
HOST_TRIPLE=""
HOST_TARGET_CPU=""
HOST_TARGET_ABI=""
HOST_FEATURE_FLAGS=()

case "${HOST_ARCH}" in
  aarch64)
    HOST_TRIPLE="aarch64-unknown-linux-gnu"
    HOST_TARGET_CPU="generic"
    HOST_TARGET_ABI="aapcs"
    HOST_FEATURE_FLAGS=(-target-feature +neon -target-feature +v8a)
    ;;
  x86_64)
    HOST_TRIPLE="x86_64-unknown-linux-gnu"
    HOST_TARGET_CPU="x86-64"
    ;;
  *)
    echo "unsupported host arch from uname -m: ${HOST_ARCH}" >&2
    exit 1
    ;;
esac

for required in "${BISHENG_BIN}" "${BISHENG_CC1_BIN}" "${CCE_LD_BIN}" "${LD_LLD_BIN}"; do
  if [[ ! -x "${required}" ]]; then
    echo "missing required tool: ${required}" >&2
    exit 1
  fi
done

readarray -t BISHENG_SYSTEM_INCLUDES < <(
  "${BISHENG_BIN}" -xc++ -E -v - </dev/null 2>&1 |
    awk '
      /#include <...> search starts here:/ {capture=1; next}
      /End of search list\./ {capture=0}
      capture && $0 ~ /^ / {sub(/^ +/, "", $0); print}
    '
)

if [[ "${#BISHENG_SYSTEM_INCLUDES[@]}" -eq 0 ]]; then
  echo "failed to discover bisheng system include directories" >&2
  exit 1
fi

CC1_INCLUDE_FLAGS=()
for inc in "${BISHENG_SYSTEM_INCLUDES[@]}"; do
  if [[ "${inc}" == */include/c++/* || "${inc}" == */backward ]]; then
    CC1_INCLUDE_FLAGS+=(-internal-isystem "${inc}")
  elif [[ "${inc}" == "/usr/include" ]]; then
    CC1_INCLUDE_FLAGS+=(-internal-externc-isystem "${inc}")
  else
    CC1_INCLUDE_FLAGS+=(-internal-isystem "${inc}")
  fi
done

OUTPUT_DIR="$(cd "$(dirname "${OUTPUT_OBJ}")" && pwd)"
BASE_NAME="$(basename "${OUTPUT_OBJ}" .o)"
HOST_STUB_OBJ="${OUTPUT_DIR}/${BASE_NAME}_host_from_llvm.o"
GENERATED_STUB_SRC="${OUTPUT_DIR}/${BASE_NAME}_stub.cpp"

{
  cat <<'EOF'
#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef AICORE
#define AICORE [aicore]
#endif
EOF
  if ! sed -n '/__global__ AICORE void /s/;$/ {}/p' "${KERNEL_STUB_SRC}"; then
    echo "failed to derive stub declarations from ${KERNEL_STUB_SRC}" >&2
    exit 1
  fi
} > "${GENERATED_STUB_SRC}"

if ! grep -q "__global__ AICORE void" "${GENERATED_STUB_SRC}"; then
  echo "no kernel declarations found in ${KERNEL_STUB_SRC}" >&2
  exit 1
fi

host_target_args=(
  -triple "${HOST_TRIPLE}"
  -target-cpu "${HOST_TARGET_CPU}"
)
if [[ -n "${HOST_TARGET_ABI}" ]]; then
  host_target_args+=(-target-abi "${HOST_TARGET_ABI}")
fi
if [[ ${#HOST_FEATURE_FLAGS[@]} -gt 0 ]]; then
  host_target_args+=("${HOST_FEATURE_FLAGS[@]}")
fi

"${BISHENG_CC1_BIN}" -cc1 \
  "${host_target_args[@]}" \
  -fcce-aicpu-legacy-launch \
  -fcce-is-host \
  -cce-launch-with-flagv2-impl \
  -fcce-aicore-arch "${AICORE_ARCH}" \
  -fcce-fatobj-compile \
  -emit-obj \
  --mrelax-relocations \
  -disable-free \
  -clear-ast-before-backend \
  -disable-llvm-verifier \
  -discard-value-names \
  -main-file-name "$(basename "${KERNEL_STUB_SRC}")" \
  -mrelocation-model pic \
  -pic-level 2 \
  -fhalf-no-semantic-interposition \
  -fenable-matrix \
  -mllvm -enable-matrix \
  -mframe-pointer=non-leaf \
  -fmath-errno \
  -ffp-contract=on \
  -fno-rounding-math \
  -mconstructor-aliases \
  -funwind-tables=2 \
  -fallow-half-arguments-and-returns \
  -mllvm -treat-scalable-fixed-error-as-warning \
  -fcoverage-compilation-dir="${ROOT_DIR}" \
  -resource-dir "${CLANG_RESOURCE_DIR}" \
  -include __clang_cce_runtime_wrapper.h \
  -D _FORTIFY_SOURCE=2 \
  -D REGISTER_BASE \
  -I "${ASCEND_HOME_PATH}/include" \
  -I "${ASCEND_HOME_PATH}/pkg_inc" \
  -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
  -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
  "${CC1_INCLUDE_FLAGS[@]}" \
  -O2 \
  -Wno-macro-redefined \
  -Wno-ignored-attributes \
  -std=c++17 \
  -fdeprecated-macro \
  -fdebug-compilation-dir="${ROOT_DIR}" \
  -ferror-limit 19 \
  -stack-protector 2 \
  -fno-signed-char \
  -fgnuc-version=4.2.1 \
  -fcxx-exceptions \
  -fexceptions \
  -vectorize-loops \
  -vectorize-slp \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  -fcce-include-aibinary "${DEVICE_OBJ}" \
  -fcce-device-module-id "${MODULE_ID}" \
  -target-feature +outline-atomics \
  -faddrsig \
  -D__GCC_HAVE_DWARF2_CFI_ASM=1 \
  -o "${HOST_STUB_OBJ}" \
  -x cce "${GENERATED_STUB_SRC}"

"${CCE_LD_BIN}" \
  "${LD_LLD_BIN}" \
  -x \
  -cce-lite-bin-module-id "${MODULE_ID}" \
  -cce-aicore-arch="${AICORE_ARCH}" \
  -r \
  -o "${OUTPUT_OBJ}" \
  -cce-stub-dir "${CCE_STUB_DIR}" \
  -cce-install-dir "$(dirname "${BISHENG_CC1_BIN}")" \
  -cce-inputs-number 1 \
  "${HOST_STUB_OBJ}"
