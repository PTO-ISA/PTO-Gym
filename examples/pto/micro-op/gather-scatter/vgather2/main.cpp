// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/gather-scatter/vgather2
// family: gather-scatter
// target_ops: pto.vgather2
// scenarios: core-f32, core-f16-u16-offsets, core-u8-u16-offsets, full-mask, non-contiguous, explicit-index-pattern, load-effect-validation, no-alias
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#ifndef TMRGSORT_HPP
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
#endif

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


void LaunchVgather2DeepMerged(float * p0, int * p1, float * p2,
                              uint16_t * p3, uint16_t * p4, uint16_t * p5,
                              uint8_t * p6, uint16_t * p7,
                              int8_t * p8, int16_t * p9,
                              void *stream);
int main() {
    size_t elemCount_v1 = 1024;
    size_t fileSize_v1 = elemCount_v1 * sizeof(float);
    size_t elemCount_v2 = 1024;
    size_t fileSize_v2 = elemCount_v2 * sizeof(int);
    size_t elemCount_v3 = 1024;
    size_t fileSize_v3 = elemCount_v3 * sizeof(float);
    size_t elemCount_v4 = 1024;
    size_t fileSize_v4 = elemCount_v4 * sizeof(uint16_t);
    size_t elemCount_v5 = 1024;
    size_t fileSize_v5 = elemCount_v5 * sizeof(uint16_t);
    size_t elemCount_v6 = 1024;
    size_t fileSize_v6 = elemCount_v6 * sizeof(uint16_t);
    size_t elemCount_v7 = 1024;
    size_t fileSize_v7 = elemCount_v7 * sizeof(uint8_t);
    size_t elemCount_v8 = 1024;
    size_t fileSize_v8 = elemCount_v8 * sizeof(uint16_t);
    size_t elemCount_v9 = 1024;
    size_t fileSize_v9 = elemCount_v9 * sizeof(int8_t);
    size_t elemCount_v10 = 1024;
    size_t fileSize_v10 = elemCount_v10 * sizeof(int16_t);
    float *v1Host = nullptr;
    float *v1Device = nullptr;
    int *v2Host = nullptr;
    int *v2Device = nullptr;
    float *v3Host = nullptr;
    float *v3Device = nullptr;
    uint16_t *v4Host = nullptr;
    uint16_t *v4Device = nullptr;
    uint16_t *v5Host = nullptr;
    uint16_t *v5Device = nullptr;
    uint16_t *v6Host = nullptr;
    uint16_t *v6Device = nullptr;
    uint8_t *v7Host = nullptr;
    uint8_t *v7Device = nullptr;
    uint16_t *v8Host = nullptr;
    uint16_t *v8Device = nullptr;
    int8_t *v9Host = nullptr;
    int8_t *v9Device = nullptr;
    int16_t *v10Host = nullptr;
    int16_t *v10Device = nullptr;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSize_v1));
    ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSize_v2));
    ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSize_v3));
    ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSize_v4));
    ACL_CHECK(aclrtMallocHost((void **)(&v5Host), fileSize_v5));
    ACL_CHECK(aclrtMallocHost((void **)(&v6Host), fileSize_v6));
    ACL_CHECK(aclrtMallocHost((void **)(&v7Host), fileSize_v7));
    ACL_CHECK(aclrtMallocHost((void **)(&v8Host), fileSize_v8));
    ACL_CHECK(aclrtMallocHost((void **)(&v9Host), fileSize_v9));
    ACL_CHECK(aclrtMallocHost((void **)(&v10Host), fileSize_v10));
    ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize_v1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize_v2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSize_v3, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSize_v4, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v5Device, fileSize_v5, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v6Device, fileSize_v6, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v7Device, fileSize_v7, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v8Device, fileSize_v8, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v9Device, fileSize_v9, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v10Device, fileSize_v10, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./v1.bin", fileSize_v1, v1Host, fileSize_v1);
    ReadFile("./v2.bin", fileSize_v2, v2Host, fileSize_v2);
    ReadFile("./v3.bin", fileSize_v3, v3Host, fileSize_v3);
    ReadFile("./v4.bin", fileSize_v4, v4Host, fileSize_v4);
    ReadFile("./v5.bin", fileSize_v5, v5Host, fileSize_v5);
    ReadFile("./v6.bin", fileSize_v6, v6Host, fileSize_v6);
    ReadFile("./v7.bin", fileSize_v7, v7Host, fileSize_v7);
    ReadFile("./v8.bin", fileSize_v8, v8Host, fileSize_v8);
    ReadFile("./v9.bin", fileSize_v9, v9Host, fileSize_v9);
    ReadFile("./v10.bin", fileSize_v10, v10Host, fileSize_v10);
    ACL_CHECK(aclrtMemcpy(v1Device, fileSize_v1, v1Host, fileSize_v1, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v2Device, fileSize_v2, v2Host, fileSize_v2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v3Device, fileSize_v3, v3Host, fileSize_v3, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v4Device, fileSize_v4, v4Host, fileSize_v4, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v5Device, fileSize_v5, v5Host, fileSize_v5, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v6Device, fileSize_v6, v6Host, fileSize_v6, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v7Device, fileSize_v7, v7Host, fileSize_v7, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v8Device, fileSize_v8, v8Host, fileSize_v8, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v9Device, fileSize_v9, v9Host, fileSize_v9, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v10Device, fileSize_v10, v10Host, fileSize_v10, ACL_MEMCPY_HOST_TO_DEVICE));
    LaunchVgather2DeepMerged(v1Device, v2Device, v3Device,
                              v4Device, v5Device, v6Device,
                              v7Device, v8Device,
                              v9Device, v10Device, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(v3Host, fileSize_v3, v3Device, fileSize_v3, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v6Host, fileSize_v6, v6Device, fileSize_v6, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v8Host, fileSize_v8, v8Device, fileSize_v8, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(v10Host, fileSize_v10, v10Device, fileSize_v10, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile("./v3.bin", v3Host, fileSize_v3);
    WriteFile("./v6.bin", v6Host, fileSize_v6);
    WriteFile("./v8.bin", v8Host, fileSize_v8);
    WriteFile("./v10.bin", v10Host, fileSize_v10);

cleanup:
    aclrtFree(v1Device);
    aclrtFree(v2Device);
    aclrtFree(v3Device);
    aclrtFree(v4Device);
    aclrtFree(v5Device);
    aclrtFree(v6Device);
    aclrtFree(v7Device);
    aclrtFree(v8Device);
    aclrtFree(v9Device);
    aclrtFree(v10Device);
    aclrtFreeHost(v1Host);
    aclrtFreeHost(v2Host);
    aclrtFreeHost(v3Host);
    aclrtFreeHost(v4Host);
    aclrtFreeHost(v5Host);
    aclrtFreeHost(v6Host);
    aclrtFreeHost(v7Host);
    aclrtFreeHost(v8Host);
    aclrtFreeHost(v9Host);
    aclrtFreeHost(v10Host);
    if (stream != nullptr) {
        const aclError _ret = aclrtDestroyStream(stream);
        if (_ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclrtDestroyStream(stream)", (int)_ret, __FILE__, __LINE__);
        }
        stream = nullptr;
    }
    if (deviceSet) {
        const aclError _ret = aclrtResetDevice(deviceId);
        if (_ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclrtResetDevice(deviceId)", (int)_ret, __FILE__, __LINE__);
        }
    }
    if (aclInited) {
        const aclError _ret = aclFinalize();
        if (_ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclFinalize()", (int)_ret, __FILE__, __LINE__);
        }
    }

    return rc;
}
