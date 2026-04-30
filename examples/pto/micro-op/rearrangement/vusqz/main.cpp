// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/rearrangement/vusqz
// family: rearrangement
// target_ops: pto.vusqz
// scenarios: predicate-driven-rearrangement, prefix-count
// -----------------------------------------------------------------------------
#include "test_common.h"
#include "acl/acl.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                  \
    do {                                                                                 \
        const aclError _ret = (expr);                                                    \
        if (_ret != ACL_SUCCESS) {                                                       \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret,   \
                         __FILE__, __LINE__);                                            \
            const char *_recent = aclGetRecentErrMsg();                                  \
            if (_recent != nullptr && _recent[0] != '\0')                                \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
            rc = 1;                                                                      \
            goto cleanup;                                                                \
        }                                                                                \
    } while (0)

void LaunchVusqz_kernel_2d(int32_t *v1, float *v2, int32_t *v3, void *stream);

int main() {
    constexpr size_t elemCount = 1024;
    size_t fileSizeV1 = elemCount * sizeof(int32_t);
    size_t fileSizeV2 = elemCount * sizeof(float);
    size_t fileSizeV3 = elemCount * sizeof(int32_t);
    int32_t *v1Host = nullptr;
    int32_t *v1Device = nullptr;
    float *v2Host = nullptr;
    float *v2Device = nullptr;
    int32_t *v3Host = nullptr;
    int32_t *v3Device = nullptr;

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

    ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSizeV1));
    ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSizeV2));
    ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSizeV3));
    ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSizeV1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSizeV2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSizeV3, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./v1.bin", fileSizeV1, v1Host, fileSizeV1);
    ReadFile("./v2.bin", fileSizeV2, v2Host, fileSizeV2);
    std::fill_n(v3Host, elemCount, 0);
    ACL_CHECK(aclrtMemcpy(v1Device, fileSizeV1, v1Host, fileSizeV1, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v2Device, fileSizeV2, v2Host, fileSizeV2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v3Device, fileSizeV3, v3Host, fileSizeV3, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchVusqz_kernel_2d(v1Device, v2Device, v3Device, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(v3Host, fileSizeV3, v3Device, fileSizeV3, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./v3.bin", v3Host, fileSizeV3);

cleanup:
    aclrtFree(v1Device);
    aclrtFree(v2Device);
    aclrtFree(v3Device);
    aclrtFreeHost(v1Host);
    aclrtFreeHost(v2Host);
    aclrtFreeHost(v3Host);
    if (stream != nullptr)
        (void)aclrtDestroyStream(stream);
    if (deviceSet)
        (void)aclrtResetDevice(deviceId);
    if (aclInited)
        (void)aclFinalize();
    return rc;
}
