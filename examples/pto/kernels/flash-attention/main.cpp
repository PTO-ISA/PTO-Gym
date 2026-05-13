// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/flash-attention
// family: kernels
// target_ops: pto.mte_gm_l1_frac, pto.mte_l1_l0a, pto.mte_l1_l0b, pto.mad,
//   pto.mte_l0c_ub, pto.mte_gm_ub, pto.mte_ub_gm, pto.vlds, pto.vcmax,
//   pto.vdup, pto.vmax, pto.vexpdif, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv,
//   pto.vsts, pto.sync.set, pto.sync.wait
// scenarios: flash-attention, cube-qk, tiled-online-softmax, q32-k32-d8
// -----------------------------------------------------------------------------
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
            if (_recent != nullptr && _recent[0] != '\0')                                        \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                     \
            rc = 1;                                                                              \
            goto cleanup;                                                                        \
        }                                                                                        \
    } while (0)

void LaunchFlash_attention_kernel_2d(float *q, float *k, float *value_t,
                                     float *out, int32_t seq, int32_t rows,
                                     void *stream);

int main() {
    constexpr int32_t rows = 32;
    constexpr int32_t seq = 32;
    constexpr int32_t headDim = 16;
    constexpr int32_t valueDim = 8;
    constexpr size_t qSize = rows * headDim * sizeof(float);
    constexpr size_t kSize = seq * headDim * sizeof(float);
    constexpr size_t valueTSize = valueDim * seq * sizeof(float);
    constexpr size_t outSize = rows * valueDim * sizeof(float);
    constexpr size_t scalarSize = sizeof(int32_t);

    float *qHost = nullptr, *kHost = nullptr, *valueTHost = nullptr, *outHost = nullptr;
    float *qDevice = nullptr, *kDevice = nullptr, *valueTDevice = nullptr, *outDevice = nullptr;
    int32_t seqHost = 0, rowsHost = 0;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;
    size_t inputSize = 0;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(envDevice);
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    inputSize = scalarSize;
    ReadFile("./v5.bin", inputSize, &seqHost, scalarSize);
    inputSize = scalarSize;
    ReadFile("./v6.bin", inputSize, &rowsHost, scalarSize);

    ACL_CHECK(aclrtMallocHost((void **)(&qHost), qSize));
    ACL_CHECK(aclrtMallocHost((void **)(&kHost), kSize));
    ACL_CHECK(aclrtMallocHost((void **)(&valueTHost), valueTSize));
    ACL_CHECK(aclrtMallocHost((void **)(&outHost), outSize));

    ACL_CHECK(aclrtMalloc((void **)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&valueTDevice, valueTSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&outDevice, outSize, ACL_MEM_MALLOC_HUGE_FIRST));

    inputSize = qSize;
    ReadFile("./v1.bin", inputSize, qHost, qSize);
    inputSize = kSize;
    ReadFile("./v2.bin", inputSize, kHost, kSize);
    inputSize = valueTSize;
    ReadFile("./v3.bin", inputSize, valueTHost, valueTSize);
    inputSize = outSize;
    ReadFile("./v4.bin", inputSize, outHost, outSize);

    ACL_CHECK(aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(valueTDevice, valueTSize, valueTHost, valueTSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(outDevice, outSize, outHost, outSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchFlash_attention_kernel_2d(qDevice, kDevice, valueTDevice, outDevice,
                                    seqHost, rowsHost, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(outHost, outSize, outDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./v4.bin", outHost, outSize);

cleanup:
    aclrtFree(qDevice); aclrtFree(kDevice); aclrtFree(valueTDevice); aclrtFree(outDevice);
    aclrtFreeHost(qHost); aclrtFreeHost(kHost); aclrtFreeHost(valueTHost); aclrtFreeHost(outHost);
    if (stream != nullptr) {
        const aclError _ret = aclrtDestroyStream(stream);
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclrtDestroyStream(stream)", (int)_ret, __FILE__, __LINE__);
    }
    if (deviceSet) {
        const aclError _ret = aclrtResetDevice(deviceId);
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclrtResetDevice(deviceId)", (int)_ret, __FILE__, __LINE__);
    }
    if (aclInited) {
        const aclError _ret = aclFinalize();
        if (_ret != ACL_SUCCESS)
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n",
                         "aclFinalize()", (int)_ret, __FILE__, __LINE__);
    }

    return rc;
}
