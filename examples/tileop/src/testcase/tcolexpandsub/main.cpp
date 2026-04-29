// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file in the compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tcolexpandsub ST
// Test cases match PTO-ISA: /home/zhoushaofan/code/pto-isa/tests/npu/a5/src/st/testcase/tcolexpandsub/
// TCOLEXPANDSUB: subtract src0 by expanded src1 (broadcast src1 first row)

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
void LaunchTCOLEXPANDSUB_fp32_6_128_1_128(float *src0, float *src1, float *dst, void *stream);
void LaunchTCOLEXPANDSUB_fp32_18_32_1_32(float *src0, float *src1, float *dst, void *stream);
void LaunchTCOLEXPANDSUB_fp16_10_256_1_256(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTCOLEXPANDSUB_fp16_12_64_1_64(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTCOLEXPANDSUB_int32_16_32_1_32(int32_t *src0, int32_t *src1, int32_t *dst, void *stream);
void LaunchTCOLEXPANDSUB_int16_16_64_1_64(int16_t *src0, int16_t *src1, int16_t *dst, void *stream);

using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      src0Rows;
    size_t      src0Cols;
    size_t      src1Rows;
    size_t      src1Cols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      validRows;
    size_t      validCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"fp32_6_128_1_128",   (LaunchFn)LaunchTCOLEXPANDSUB_fp32_6_128_1_128,   6, 128, 1, 128, 6, 128, 6, 128, sizeof(float)},
    {"fp32_18_32_1_32",    (LaunchFn)LaunchTCOLEXPANDSUB_fp32_18_32_1_32,   18,  32, 1,  32,18,  32,18,  32, sizeof(float)},
    {"fp16_10_256_1_256",  (LaunchFn)LaunchTCOLEXPANDSUB_fp16_10_256_1_256, 10, 256, 1, 256,10, 256,10, 256, sizeof(uint16_t)},
    {"fp16_12_64_1_64",    (LaunchFn)LaunchTCOLEXPANDSUB_fp16_12_64_1_64,   12,  64, 1,  64,12,  64,12,  64, sizeof(uint16_t)},
    {"int32_16_32_1_32",   (LaunchFn)LaunchTCOLEXPANDSUB_int32_16_32_1_32,  16,  32, 1,  32, 16,  32, 16,  32, sizeof(int32_t)},
    {"int16_16_64_1_64",   (LaunchFn)LaunchTCOLEXPANDSUB_int16_16_64_1_64,  16,  64, 1,  64, 16,  64, 16,  64, sizeof(int16_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t src0ElemCount = tc.src0Rows * tc.src0Cols;
    const size_t src1ElemCount = tc.src1Rows * tc.src1Cols;
    const size_t dstElemCount  = tc.dstRows * tc.dstCols;
    const size_t src0FileSize  = src0ElemCount * tc.elemSize;
    const size_t src1FileSize  = src1ElemCount * tc.elemSize;
    const size_t dstFileSize   = dstElemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (src0=%zux%zu, src1=%zux%zu -> dst=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.src0Rows, tc.src0Cols, tc.src1Rows, tc.src1Cols, tc.dstRows, tc.dstCols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;
    size_t actualSrc0FileSize = src0FileSize;
    size_t actualSrc1FileSize = src1FileSize;

    void *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);

    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input0.bin").c_str(), actualSrc0FileSize, src0Host, src0FileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0 && !ReadFile((caseDir + "/input1.bin").c_str(), actualSrc1FileSize, src1Host, src1FileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(src0Device, src1Device, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (src0Device != nullptr)
        aclrtFree(src0Device);
    if (src1Device != nullptr)
        aclrtFree(src1Device);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (src0Host != nullptr)
        aclrtFreeHost(src0Host);
    if (src1Host != nullptr)
        aclrtFreeHost(src1Host);
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