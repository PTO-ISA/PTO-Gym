// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

void LaunchTMATMUL_f16_16x16x16(uint16_t *a, uint16_t *b, float *c, void *stream);

using LaunchFn = void (*)(uint16_t *, uint16_t *, float *, void *);

struct TestCase {
    const char *name;
    LaunchFn launch;
    size_t lhsRows;
    size_t lhsCols;
    size_t rhsRows;
    size_t rhsCols;
    size_t outRows;
    size_t outCols;
};

static const TestCase kCases[] = {
    {"f16_16x16x16", LaunchTMATMUL_f16_16x16x16, 16, 16, 16, 16, 16, 16},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t lhsElems = tc.lhsRows * tc.lhsCols;
    const size_t rhsElems = tc.rhsRows * tc.rhsCols;
    const size_t outElems = tc.outRows * tc.outCols;
    const size_t lhsBytes = lhsElems * sizeof(uint16_t);
    const size_t rhsBytes = rhsElems * sizeof(uint16_t);
    const size_t outBytes = outElems * sizeof(float);
    size_t lhsFileSize = lhsBytes;
    size_t rhsFileSize = rhsBytes;

    std::printf(
        "[INFO] === case: %s (lhs=%zux%zu, rhs=%zux%zu, out=%zux%zu) ===\n",
        tc.name,
        tc.lhsRows,
        tc.lhsCols,
        tc.rhsRows,
        tc.rhsCols,
        tc.outRows,
        tc.outCols
    );

    std::string caseDir = std::string("./") + tc.name;

    void *lhsHost = nullptr;
    void *rhsHost = nullptr;
    void *outHost = nullptr;
    void *lhsDevice = nullptr;
    void *rhsDevice = nullptr;
    void *outDevice = nullptr;

    aclrtMallocHost(&lhsHost, lhsBytes);
    aclrtMallocHost(&rhsHost, rhsBytes);
    aclrtMallocHost(&outHost, outBytes);

    aclrtMalloc(&lhsDevice, lhsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&rhsDevice, rhsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), lhsFileSize, lhsHost, lhsBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), rhsFileSize, rhsHost, rhsBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(lhsDevice, lhsBytes, lhsHost, lhsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(rhsDevice, rhsBytes, rhsHost, rhsBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(
            static_cast<uint16_t *>(lhsDevice),
            static_cast<uint16_t *>(rhsDevice),
            static_cast<float *>(outDevice),
            stream
        );

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outHost, outBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (lhsDevice != nullptr)
        aclrtFree(lhsDevice);
    if (rhsDevice != nullptr)
        aclrtFree(rhsDevice);
    if (outDevice != nullptr)
        aclrtFree(outDevice);
    if (lhsHost != nullptr)
        aclrtFreeHost(lhsHost);
    if (rhsHost != nullptr)
        aclrtFreeHost(rhsHost);
    if (outHost != nullptr)
        aclrtFreeHost(outHost);

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
