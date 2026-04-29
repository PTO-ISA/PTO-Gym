// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trowexpand ST — row broadcast operation.
// Supports multiple data types: f32, f16, i8

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
// f32
void LaunchTROWEXPAND_f32_16x128(float *src, float *dst, void *stream);
void LaunchTROWEXPAND_f32_16x127(float *src, float *dst, void *stream);
// f16
void LaunchTROWEXPAND_f16_16x512(void *src, void *dst, void *stream);
void LaunchTROWEXPAND_f16_16x511(void *src, void *dst, void *stream);
// i8
void LaunchTROWEXPAND_i8_16x256(void *src, void *dst, void *stream);
void LaunchTROWEXPAND_i8_16x255(void *src, void *dst, void *stream);

// Generic launch function type
using LaunchFn = void (*)(void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      srcRows;
    size_t      srcCols;       // srcCols = 32/sizeof(dtype) for alignment
    size_t      dstRows;
    size_t      dstCols;
    size_t      dstValidCols;  // effective output columns
    size_t      elemSize;      // bytes per element
};

static const TestCase kCases[] = {
    // f32: srcCols=8 (32/4), dstCols=128, dstValidCols=128 or 127
    {"f32_16x128", (LaunchFn)LaunchTROWEXPAND_f32_16x128, 16, 8, 16, 128, 128, sizeof(float)},
    {"f32_16x127", (LaunchFn)LaunchTROWEXPAND_f32_16x127, 16, 8, 16, 128, 127, sizeof(float)},
    // f16: srcCols=16 (32/2), dstCols=512, dstValidCols=512 or 511
    {"f16_16x512", LaunchTROWEXPAND_f16_16x512, 16, 16, 16, 512, 512, sizeof(uint16_t)},
    {"f16_16x511", LaunchTROWEXPAND_f16_16x511, 16, 16, 16, 512, 511, sizeof(uint16_t)},
    // i8: srcCols=32 (32/1), dstCols=256, dstValidCols=256 or 255
    {"i8_16x256", LaunchTROWEXPAND_i8_16x256, 16, 32, 16, 256, 256, sizeof(int8_t)},
    {"i8_16x255", LaunchTROWEXPAND_i8_16x255, 16, 32, 16, 256, 255, sizeof(int8_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t srcFileSize  = tc.srcRows * tc.srcCols * tc.elemSize;
    const size_t dstFileSize = tc.dstRows * tc.dstCols * tc.elemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu, valid_cols=%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols, tc.dstValidCols);

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), srcFileSize, srcHost, srcFileSize)) {
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