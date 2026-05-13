#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Script to generate launch.cpp & main.cpp"""

import numpy as np
import cases
from cases import bfloat16

_DTYPE_TO_CPP = {
    np.float32: "float",
    np.float16: "uint16_t",
    bfloat16: "uint16_t",
    np.int8: "int8_t",
    np.uint8: "uint8_t",
    np.int16: "int16_t",
    "si16": "int16_t",
    np.uint16: "uint16_t",
    np.int32: "int32_t",
    np.uint32: "uint32_t",
    np.int64: "int64_t",
    np.uint64: "uint64_t",
}

def gen_launch():
    lines = [
        "// Copyright (c) 2026 Huawei Technologies Co., Ltd.",
        "// This program is free software, you can redistribute it and/or modify it under the terms and conditions of",
        '// CANN Open Software License Agreement Version 2.0 (the "License").',
        "// Please refer to the License for details. You may not use this file except in compliance with the License.",
        '// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,',
        "// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.",
        "// See LICENSE in the root of the software repository for the full text of the License.",
        "",
        "#include <stdint.h>",
        "",
        "#ifndef AICORE",
        "#define AICORE [aicore]",
        "#endif",
        "",
    ]
    
    extern_decls = []
    launch_funcs = []
    
    for c in cases.CASES:
        name = c["name"]
        src_cpp = _DTYPE_TO_CPP.get(c["src_dtype"], "float")
        dst_cpp = _DTYPE_TO_CPP.get(c["dst_dtype"], "float")
        
        extern_decls.append(f'extern "C" __global__ AICORE void TCVT_{name}(__gm__ {src_cpp} *src, __gm__ {dst_cpp} *dst);')
        launch_funcs.append(f"void LaunchTCVT_{name}(void *src, void *dst, void *stream) {{")
        launch_funcs.append(f"    TCVT_{name}<<<1, nullptr, stream>>>((__gm__ {src_cpp} *)src, (__gm__ {dst_cpp} *)dst);")
        launch_funcs.append("}")
        launch_funcs.append("")
    
    lines.extend(extern_decls)
    lines.append("")
    lines.extend(launch_funcs)
    
    return "\n".join(lines)

def gen_main():
    lines = [
        "// Copyright (c) 2026 Huawei Technologies Co., Ltd.",
        "// This program is free software, you can redistribute it and/or modify it under the terms and conditions of",
        '// CANN Open Software License Agreement Version 2.0 (the "License").',
        "// Please refer to the License for details. You may not use this file except in compliance with the License.",
        '// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,',
        "// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.",
        "// See LICENSE in the root of the software repository for the full text of the License.",
        "",
        '#include "acl/acl.h"',
        '#include "test_common.h"',
        "#include <cstddef>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <string>",
        "",
        "using namespace PtoTestCommon;",
        "",
    ]
    
    decls = []
    for c in cases.CASES:
        decls.append(f"void LaunchTCVT_{c['name']}(void *src, void *dst, void *stream);")
    
    lines.extend(decls)
    lines.extend([
        "",
        "using LaunchFn = void (*)(void *, void *, void *);",
        "",
        "struct TestCase {",
        "    const char *name;",
        "    LaunchFn    launch;",
        "    size_t      srcRows;",
        "    size_t      srcCols;",
        "    size_t      dstRows;",
        "    size_t      dstCols;",
        "    size_t      srcElemSize;",
        "    size_t      dstElemSize;",
        "};",
        "",
        "static const TestCase kCases[] = {",
    ])
    
    case_entries = []
    for c in cases.CASES:
        name = c["name"]
        src_cpp = _DTYPE_TO_CPP.get(c["src_dtype"], "float")
        dst_cpp = _DTYPE_TO_CPP.get(c["dst_dtype"], "float")
        rows, cols = c["shape"]
        case_entries.append(f'    {{"{name}", LaunchTCVT_{name}, {rows}, {cols}, {rows}, {cols}, sizeof({src_cpp}), sizeof({dst_cpp})}},')
    
    lines.extend(case_entries)
    lines.extend([
        "};",
        "static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);",
        "",
    ])
    
    # RunCase 和 main 函数保持不变
    lines.extend([
        "static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {",
        "    (void)deviceId;",
        "    int rc = 0;",
        "    const size_t srcElemCount = tc.srcRows * tc.srcCols;",
        "    const size_t dstElemCount = tc.dstRows * tc.dstCols;",
        "    size_t srcFileSize = srcElemCount * tc.srcElemSize;",
        "    size_t dstFileSize = dstElemCount * tc.dstElemSize;",
        "",
        '    std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu) ===\\n",',
        "                tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols);",
        "",
        '    std::string caseDir = std::string("./") + tc.name;',
        "",
        "    void *srcHost = nullptr;",
        "    void *dstHost = nullptr;",
        "    void *srcDevice = nullptr;",
        "    void *dstDevice = nullptr;",
        "",
        "    aclrtMallocHost(&srcHost, srcFileSize);",
        "    aclrtMallocHost(&dstHost, dstFileSize);",
        "",
        "    aclrtMalloc(&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);",
        "    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);",
        "",
        '    if (!ReadFile((caseDir + "/input.bin").c_str(), srcFileSize, srcHost, srcFileSize)) {',
        '        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\\n", caseDir.c_str());',
        "        rc = 1;",
        "    }",
        "",
        "    if (rc == 0) {",
        "        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);",
        "        tc.launch(srcDevice, dstDevice, stream);",
        "        aclrtSynchronizeStream(stream);",
        "        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);",
        "    }",
        "",
        '    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {',
        '        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\\n", caseDir.c_str());',
        "        rc = 1;",
        "    }",
        "",
        "    if (srcDevice != nullptr)",
        "        aclrtFree(srcDevice);",
        "    if (dstDevice != nullptr)",
        "        aclrtFree(dstDevice);",
        "    if (srcHost != nullptr)",
        "        aclrtFreeHost(srcHost);",
        "    if (dstHost != nullptr)",
        "        aclrtFreeHost(dstHost);",
        "",
        "    if (rc == 0)",
        '        std::printf("[INFO] case %s done\\n", tc.name);',
        "    return rc;",
        "}",
        "",
        "int main(int argc, char *argv[]) {",
        "    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;",
        "",
        "    int rc = 0;",
        "    int deviceId = 0;",
        "    aclrtStream stream = nullptr;",
        "",
        "    aclInit(nullptr);",
        '    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {',
        "        deviceId = std::atoi(envDevice);",
        "    }",
        "    aclrtSetDevice(deviceId);",
        "    aclrtCreateStream(&stream);",
        "",
        "    for (size_t i = 0; i < kNumCases; ++i) {",
        "        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {",
        "            continue;",
        "        }",
        "        int ret = RunCase(kCases[i], deviceId, stream);",
        "        if (ret != 0) {",
        '            std::fprintf(stderr, "[ERROR] case %s failed\\n", kCases[i].name);',
        "            rc = 1;",
        "            break;",
        "        }",
        "    }",
        "",
        "    if (stream != nullptr)",
        "        aclrtDestroyStream(stream);",
        "    aclrtResetDevice(deviceId);",
        "    aclFinalize();",
        "",
        "    return rc;",
        "}",
        ""
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    from pathlib import Path
    HERE = Path(__file__).parent

    with open(HERE / "launch.cpp", "w") as f:
        f.write(gen_launch())
    print(f"Generated {(HERE / 'launch.cpp').as_posix()!r}")
    
    with open(HERE / "main.cpp", "w") as f:
        f.write(gen_main())
    print(f"Generated {(HERE / 'main.cpp').as_posix()!r}")