// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trelu ST — case-table driven.
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
void LaunchTRELU_int32_64x64(int32_t *input, int32_t *output, void *stream);
void LaunchTRELU_f16_64x64_v60x60(uint16_t *input, uint16_t *output, void *stream);
void LaunchTRELU_f32_64x64_v60x60(float *input, float *output, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(void *, void *, void *);
    size_t      rows;
    size_t      cols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"int32_64x64",             (void (*)(void*, void*, void*))LaunchTRELU_int32_64x64,     64, 64, sizeof(int32_t)},
    {"f16_64x64_valid_60x60",   (void (*)(void*, void*, void*))LaunchTRELU_f16_64x64_v60x60, 60, 60, sizeof(uint16_t)},
    {"f32_64x64_valid_60x60",   (void (*)(void*, void*, void*))LaunchTRELU_f32_64x64_v60x60, 60, 60, sizeof(float)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    size_t fileSize  = elemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols);

    std::string caseDir = std::string("./") + tc.name;

    void *inputHost = nullptr, *outputHost = nullptr;
    void *inputDevice = nullptr, *outputDevice = nullptr;

    aclrtMallocHost(&inputHost, fileSize);
    aclrtMallocHost(&outputHost, fileSize);

    aclrtMalloc(&inputDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outputDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), fileSize, inputHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(inputDevice, fileSize, inputHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(inputDevice, outputDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outputHost, fileSize, outputDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outputHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (inputDevice != nullptr)
        aclrtFree(inputDevice);
    if (outputDevice != nullptr)
        aclrtFree(outputDevice);
    if (inputHost != nullptr)
        aclrtFreeHost(inputHost);
    if (outputHost != nullptr)
        aclrtFreeHost(outputHost);

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