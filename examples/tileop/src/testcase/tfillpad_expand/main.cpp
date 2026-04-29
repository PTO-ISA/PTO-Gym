// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tfillpad_expand ST — case-table driven.
// Each case launches a different kernel variant, reads/writes from per-case subdirectory.
// Numerical comparison is done externally by compare.py.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

// Kernel launch wrappers (defined in launch.cpp)
void LaunchTFILLPAD_EXPAND_u16_260x32_src_259x7(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTFILLPAD_EXPAND_s8_260x64_src_259x7(int8_t *src, int8_t *dst, void *stream);

enum class DataType { U16, S8 };

struct TestCase {
    const char *name;
    DataType    dtype;
    void (*launch)(void *, void *, void *);  // Generic launch function pointer
    size_t      srcRows;
    size_t      srcCols;
    size_t      srcValidRows;
    size_t      srcValidCols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      dstValidRows;
    size_t      dstValidCols;
    size_t      elemSize;
};

// Helper to wrap type-specific launch functions
template<typename T>
void wrapLaunch(void *src, void *dst, void *stream, void (*fn)(T *, T *, void *)) {
    fn((T *)src, (T *)dst, stream);
}

static const TestCase kCases[] = {
    // ========== uint16 case (C++ case 8) ==========
    {"u16_260x32_src_259x7", DataType::U16,
     [](void *src, void *dst, void *stream) { wrapLaunch<uint16_t>(src, dst, stream, LaunchTFILLPAD_EXPAND_u16_260x32_src_259x7); },
     260, 32, 259, 7, 260, 32, 260, 32, sizeof(uint16_t)},

    // ========== int8 case (C++ case 9) ==========
    {"s8_260x64_src_259x7", DataType::S8,
     [](void *src, void *dst, void *stream) { wrapLaunch<int8_t>(src, dst, stream, LaunchTFILLPAD_EXPAND_s8_260x64_src_259x7); },
     260, 64, 259, 7, 260, 64, 260, 64, sizeof(int8_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t srcElemCount = tc.srcRows * tc.srcCols;
    size_t dstElemCount = tc.dstRows * tc.dstCols;
    size_t srcFileSize  = srcElemCount * tc.elemSize;
    size_t dstFileSize  = dstElemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu valid=%zux%zu -> dst=%zux%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.srcValidRows, tc.srcValidCols, tc.dstRows, tc.dstCols);

    std::string caseDir = std::string("./") + tc.name;
    size_t inputFileSize = srcFileSize;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&srcHost, srcFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);

    aclrtMalloc(&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), inputFileSize, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr)
        aclrtFree(srcDevice);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (srcHost != nullptr)
        aclrtFreeHost(srcHost);
    if (dstHost != nullptr)
        aclrtFreeHost(dstHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}