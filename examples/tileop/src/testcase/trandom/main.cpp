// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trandom ST — case-table driven.
// Each case launches a different kernel variant, reads key/counter and writes output.
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
void LaunchTRANDOM_int32_4x256(uint32_t *key, uint32_t *counter, uint32_t *output, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(uint32_t *, uint32_t *, uint32_t *, void *);
    size_t      rows;
    size_t      cols;
};

static const TestCase kCases[] = {
    {"int32_4x256", LaunchTRANDOM_int32_4x256, 4, 256},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t outputSize = elemCount * sizeof(uint32_t);
    size_t keySize = 2 * sizeof(uint32_t);
    size_t counterSize = 4 * sizeof(uint32_t);

    std::printf("[INFO] === case: %s (shape=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols);

    std::string caseDir = std::string("./") + tc.name;

    void *keyHost = nullptr, *counterHost = nullptr, *outputHost = nullptr;
    void *keyDevice = nullptr, *counterDevice = nullptr, *outputDevice = nullptr;

    aclrtMallocHost(&keyHost, keySize);
    aclrtMallocHost(&counterHost, counterSize);
    aclrtMallocHost(&outputHost, outputSize);

    aclrtMalloc(&keyDevice, keySize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&counterDevice, counterSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/key.bin").c_str(), keySize, keyHost, keySize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/key.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (!ReadFile((caseDir + "/counter.bin").c_str(), counterSize, counterHost, counterSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/counter.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(keyDevice, keySize, keyHost, keySize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(counterDevice, counterSize, counterHost, counterSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch((uint32_t *)keyDevice, (uint32_t *)counterDevice, (uint32_t *)outputDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outputHost, outputSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (keyDevice != nullptr)
        aclrtFree(keyDevice);
    if (counterDevice != nullptr)
        aclrtFree(counterDevice);
    if (outputDevice != nullptr)
        aclrtFree(outputDevice);
    if (keyHost != nullptr)
        aclrtFreeHost(keyHost);
    if (counterHost != nullptr)
        aclrtFreeHost(counterHost);
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