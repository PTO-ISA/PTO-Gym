// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/vdup-lane
// family: materialization-predicate
// target_ops: pto.vdup
// scenarios: core-f32, vector-input, lowest-highest
// -----------------------------------------------------------------------------
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError _ret = (expr);                                                            \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            const char *_recent = aclGetRecentErrMsg();                                          \
            if (_recent != nullptr && _recent[0] != '\0') {                                      \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                     \
            }                                                                                    \
            rc = 1;                                                                              \
            goto cleanup;                                                                        \
        }                                                                                        \
    } while (0)

void LaunchVdup_lane_kernel_2d(float *src, float *outLow, float *outHigh, void *stream);

int main() {
    size_t elemCount = 1024;
    size_t fileSize = elemCount * sizeof(float);
    float *srcHost = nullptr;
    float *outLowHost = nullptr;
    float *outHighHost = nullptr;
    float *srcDevice = nullptr;
    float *outLowDevice = nullptr;
    float *outHighDevice = nullptr;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(envDevice);
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost((void **)(&srcHost), fileSize));
    ACL_CHECK(aclrtMallocHost((void **)(&outLowHost), fileSize));
    ACL_CHECK(aclrtMallocHost((void **)(&outHighHost), fileSize));
    ACL_CHECK(aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&outLowDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&outHighDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./src.bin", fileSize, srcHost, fileSize);
    ACL_CHECK(aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(outLowDevice, fileSize, 0, fileSize));
    ACL_CHECK(aclrtMemset(outHighDevice, fileSize, 0, fileSize));

    LaunchVdup_lane_kernel_2d(srcDevice, outLowDevice, outHighDevice, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(outLowHost, fileSize, outLowDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(outHighHost, fileSize, outHighDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./out_low.bin", outLowHost, fileSize);
    WriteFile("./out_high.bin", outHighHost, fileSize);

cleanup:
    aclrtFree(srcDevice);
    aclrtFree(outLowDevice);
    aclrtFree(outHighDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(outLowHost);
    aclrtFreeHost(outHighHost);
    if (stream != nullptr) {
        const aclError ret = aclrtDestroyStream(stream);
        if (ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] aclrtDestroyStream(stream) failed: %d (%s:%d)\n",
                         (int)ret, __FILE__, __LINE__);
    }
    if (deviceSet) {
        const aclError ret = aclrtResetDevice(deviceId);
        if (ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] aclrtResetDevice(deviceId) failed: %d (%s:%d)\n",
                         (int)ret, __FILE__, __LINE__);
    }
    if (aclInited) {
        const aclError ret = aclFinalize();
        if (ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] aclFinalize() failed: %d (%s:%d)\n",
                         (int)ret, __FILE__, __LINE__);
    }

    return rc;
}
