# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT DEFINED PTOAS_BIN OR NOT DEFINED PTO_SRC OR NOT DEFINED KERNEL_LL)
    message(FATAL_ERROR "PTOAS_BIN, PTO_SRC, and KERNEL_LL must be provided")
endif()

get_filename_component(KERNEL_LL_DIR "${KERNEL_LL}" DIRECTORY)
file(MAKE_DIRECTORY "${KERNEL_LL_DIR}")

if(NOT DEFINED PTOAS_ENABLE_INSERT_SYNC)
    set(PTOAS_ENABLE_INSERT_SYNC ON)
endif()

set(PTOAS_COMMAND
    "${PTOAS_BIN}"
    --pto-arch=a5
)

if(DEFINED PTOAS_PTO_LEVEL AND NOT PTOAS_PTO_LEVEL STREQUAL "")
    list(APPEND PTOAS_COMMAND "--pto-level=${PTOAS_PTO_LEVEL}")
endif()

list(APPEND PTOAS_COMMAND --pto-backend=vpto)

if(PTOAS_ENABLE_INSERT_SYNC)
    list(APPEND PTOAS_COMMAND --enable-insert-sync)
endif()

list(APPEND PTOAS_COMMAND
    --enable-tile-op-expand
    --vpto-emit-hivm-llvm
    "${PTO_SRC}"
    -o
    -
)

execute_process(
    COMMAND ${PTOAS_COMMAND}
    OUTPUT_FILE "${KERNEL_LL}"
    ERROR_VARIABLE PTOAS_STDERR
    RESULT_VARIABLE PTOAS_RESULT
)

if(NOT PTOAS_RESULT EQUAL 0)
    file(REMOVE "${KERNEL_LL}")
    string(STRIP "${PTOAS_STDERR}" PTOAS_STDERR)
    if(PTOAS_STDERR)
        message(FATAL_ERROR "ptoas failed while generating ${KERNEL_LL}:\n${PTOAS_STDERR}")
    endif()
    message(FATAL_ERROR "ptoas failed while generating ${KERNEL_LL}")
endif()

if(NOT EXISTS "${KERNEL_LL}")
    message(FATAL_ERROR "ptoas completed without producing ${KERNEL_LL}")
endif()

file(SIZE "${KERNEL_LL}" KERNEL_LL_SIZE)
if(KERNEL_LL_SIZE EQUAL 0)
    file(REMOVE "${KERNEL_LL}")
    string(STRIP "${PTOAS_STDERR}" PTOAS_STDERR)
    if(PTOAS_STDERR)
        message(FATAL_ERROR
            "ptoas produced empty LLVM IR for ${PTO_SRC}:\n${PTOAS_STDERR}")
    endif()
    message(FATAL_ERROR "ptoas produced empty LLVM IR for ${PTO_SRC}")
endif()
