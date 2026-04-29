// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

#ifndef TMRGSORT_HPP
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError _ret = (expr);                                                            \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            const char *_recent = aclGetRecentErrMsg();                                          \
            if (_recent != nullptr && _recent[0] != '\0')                                        \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                     \
            rc = 1;                                                                              \
            goto cleanup;                                                                        \
        }                                                                                        \
    } while (0)

void LaunchSOFTMAX_f32_rows24_seq73(float *v1, float *v2, float *v3,
                                    float *v4, float *v5, float *v6,
                                    float *v7, int32_t v8, int32_t v9,
                                    void *stream);

using LaunchFn = void (*)(float *, float *, float *, float *, float *, float *,
                          float *, int32_t, int32_t, void *);

struct TestCase {
    const char *name;
    LaunchFn launch;
    size_t rows;
    size_t cols;
};

static const TestCase kCases[] = {
    {"f32_rows24_seq73", LaunchSOFTMAX_f32_rows24_seq73, 24, 128},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, aclrtStream stream) {
    const size_t scalarBytes = sizeof(int32_t);
    const size_t stateElems = tc.rows;
    const size_t outElems = tc.rows * tc.cols;
    const size_t stateBytes = stateElems * sizeof(float);
    const size_t outBytes = outElems * sizeof(float);
    std::string caseDir = std::string("./") + tc.name;

    float *v1Host = nullptr, *v2Host = nullptr, *v3Host = nullptr;
    float *v4Host = nullptr, *v5Host = nullptr, *v6Host = nullptr, *v7Host = nullptr;
    float *v1Device = nullptr, *v2Device = nullptr, *v3Device = nullptr;
    float *v4Device = nullptr, *v5Device = nullptr, *v6Device = nullptr, *v7Device = nullptr;
    int32_t seqHost = 0;
    int32_t rowsHost = 0;
    size_t fileSize = 0;
    int rc = 0;

    std::printf("[INFO] === case: %s (rows=%zu, cols=%zu) ===\n",
                tc.name, tc.rows, tc.cols);

    if (!ReadFile(caseDir + "/v8.bin", fileSize, &seqHost, scalarBytes) ||
        !ReadFile(caseDir + "/v9.bin", fileSize, &rowsHost, scalarBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read scalar inputs for %s\n", tc.name);
        return 1;
    }

    ACL_CHECK(aclrtMallocHost((void **)(&v1Host), stateBytes));
    ACL_CHECK(aclrtMallocHost((void **)(&v2Host), stateBytes));
    ACL_CHECK(aclrtMallocHost((void **)(&v3Host), outBytes));
    ACL_CHECK(aclrtMallocHost((void **)(&v4Host), stateBytes));
    ACL_CHECK(aclrtMallocHost((void **)(&v5Host), stateBytes));
    ACL_CHECK(aclrtMallocHost((void **)(&v6Host), stateBytes));
    ACL_CHECK(aclrtMallocHost((void **)(&v7Host), outBytes));

    ACL_CHECK(aclrtMalloc((void **)&v1Device, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v3Device, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v4Device, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v5Device, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v6Device, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v7Device, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    if (!ReadFile(caseDir + "/v1.bin", fileSize, v1Host, stateBytes) ||
        !ReadFile(caseDir + "/v2.bin", fileSize, v2Host, stateBytes) ||
        !ReadFile(caseDir + "/v3.bin", fileSize, v3Host, outBytes) ||
        !ReadFile(caseDir + "/v4.bin", fileSize, v4Host, stateBytes) ||
        !ReadFile(caseDir + "/v5.bin", fileSize, v5Host, stateBytes) ||
        !ReadFile(caseDir + "/v6.bin", fileSize, v6Host, stateBytes) ||
        !ReadFile(caseDir + "/v7.bin", fileSize, v7Host, outBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read tensor inputs for %s\n", tc.name);
        rc = 1;
        goto cleanup;
    }

    ACL_CHECK(aclrtMemcpy(v1Device, stateBytes, v1Host, stateBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v2Device, stateBytes, v2Host, stateBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v3Device, outBytes, v3Host, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v4Device, stateBytes, v4Host, stateBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v5Device, stateBytes, v5Host, stateBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v6Device, stateBytes, v6Host, stateBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v7Device, outBytes, v7Host, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    tc.launch(v1Device, v2Device, v3Device, v4Device, v5Device, v6Device,
              v7Device, seqHost, rowsHost, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(v4Host, stateBytes, v4Device, stateBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v5Host, stateBytes, v5Device, stateBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v6Host, stateBytes, v6Device, stateBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v7Host, outBytes, v7Device, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    if (!WriteFile(caseDir + "/v4.bin", v4Host, stateBytes) ||
        !WriteFile(caseDir + "/v5.bin", v5Host, stateBytes) ||
        !WriteFile(caseDir + "/v6.bin", v6Host, stateBytes) ||
        !WriteFile(caseDir + "/v7.bin", v7Host, outBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write outputs for %s\n", tc.name);
        rc = 1;
    }

cleanup:
    if (v1Device != nullptr) aclrtFree(v1Device);
    if (v2Device != nullptr) aclrtFree(v2Device);
    if (v3Device != nullptr) aclrtFree(v3Device);
    if (v4Device != nullptr) aclrtFree(v4Device);
    if (v5Device != nullptr) aclrtFree(v5Device);
    if (v6Device != nullptr) aclrtFree(v6Device);
    if (v7Device != nullptr) aclrtFree(v7Device);
    if (v1Host != nullptr) aclrtFreeHost(v1Host);
    if (v2Host != nullptr) aclrtFreeHost(v2Host);
    if (v3Host != nullptr) aclrtFreeHost(v3Host);
    if (v4Host != nullptr) aclrtFreeHost(v4Host);
    if (v5Host != nullptr) aclrtFreeHost(v5Host);
    if (v6Host != nullptr) aclrtFreeHost(v6Host);
    if (v7Host != nullptr) aclrtFreeHost(v7Host);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;
    bool matchedCase = (caseFilter == nullptr);
    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(envDevice);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0)
            continue;
        matchedCase = true;
        if (RunCase(kCases[i], stream) != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (!matchedCase) {
        std::fprintf(stderr, "[ERROR] unknown case filter: %s\n", caseFilter);
        rc = 1;
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
