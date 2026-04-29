// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tshls ST — case-table driven.
// tshls: dst = src << scalar (single input + scalar, left shift).
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
void LaunchTSHLS_i32_32x64(int32_t *src, int32_t *dst, void *stream);
void LaunchTSHLS_i16_63x64(int16_t *src, int16_t *dst, void *stream);
void LaunchTSHLS_i32_31x128(int32_t *src, int32_t *dst, void *stream);
void LaunchTSHLS_i16_15x192(int16_t *src, int16_t *dst, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(void *, void *, void *);  // src, dst, stream
    size_t      rows;       // allocated tile rows
    size_t      cols;       // allocated tile cols
    size_t      validRows;  // effective computation rows  (<= rows)
    size_t      validCols;  // effective computation cols  (<= cols)
    size_t      elemSize;   // bytes per element
};

static const TestCase kCases[] = {
    {"i32_32x64",   (void (*)(void*,void*,void*))LaunchTSHLS_i32_32x64,   32,  64,  32,  64,  sizeof(int32_t)},
    {"i16_63x64",   (void (*)(void*,void*,void*))LaunchTSHLS_i16_63x64,   63,  64,  63,  64,  sizeof(int16_t)},
    {"i32_31x128",  (void (*)(void*,void*,void*))LaunchTSHLS_i32_31x128,  31,  128, 31,  128, sizeof(int32_t)},
    {"i16_15x192",  (void (*)(void*,void*,void*))LaunchTSHLS_i16_15x192,  15,  192, 15,  192, sizeof(int16_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t fileSize  = elemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols);

    // Per-case data directory
    std::string caseDir = std::string("./") + tc.name;
    size_t srcFileSize = fileSize;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&srcHost, fileSize);
    aclrtMallocHost(&dstHost, fileSize);

    aclrtMalloc(&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), srcFileSize, srcHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSize)) {
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
    // Optional case filter: ./tshls [case_name]
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
