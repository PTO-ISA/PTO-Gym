// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tcolexpand ST
// Test cases match PTO-ISA: /home/zhoushaofan/code/pto-isa/tests/npu/a5/src/st/testcase/tcolexpand/
// TCOLEXPAND: expand src first row to dst all rows by broadcasting

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
void LaunchTCOLEXPAND_half_1_16_512_512(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTCOLEXPAND_int8_2_32_256_255(int8_t *src, int8_t *dst, void *stream);
void LaunchTCOLEXPAND_float_1_8_128_63(float *src, float *dst, void *stream);
void LaunchTCOLEXPAND_half_1_33_512_512(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTCOLEXPAND_int8_2_17_256_44(int8_t *src, int8_t *dst, void *stream);
void LaunchTCOLEXPAND_float_1_54_64_63(float *src, float *dst, void *stream);

using LaunchFn = void (*)(void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      srcRows;
    size_t      srcCols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      validRows;
    size_t      validCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"half_1_16_512_512",   (LaunchFn)LaunchTCOLEXPAND_half_1_16_512_512,   1, 512, 16, 512, 16, 512, sizeof(uint16_t)},
    {"int8_2_32_256_255",   (LaunchFn)LaunchTCOLEXPAND_int8_2_32_256_255,   2, 256, 32, 256, 32, 255, sizeof(int8_t)},
    {"float_1_8_128_63",    (LaunchFn)LaunchTCOLEXPAND_float_1_8_128_63,    1, 128,  8, 128,  8,  63, sizeof(float)},
    {"half_1_33_512_512",   (LaunchFn)LaunchTCOLEXPAND_half_1_33_512_512,   1, 512, 33, 512, 33, 512, sizeof(uint16_t)},
    {"int8_2_17_256_44",    (LaunchFn)LaunchTCOLEXPAND_int8_2_17_256_44,    2, 256, 17, 256, 17,  44, sizeof(int8_t)},
    {"float_1_54_64_63",    (LaunchFn)LaunchTCOLEXPAND_float_1_54_64_63,    1,  64, 54,  64, 54,  63, sizeof(float)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcElemCount = tc.srcRows * tc.srcCols;
    const size_t dstElemCount = tc.dstRows * tc.dstCols;
    const size_t srcFileSize  = srcElemCount * tc.elemSize;
    const size_t dstFileSize  = dstElemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu -> dst=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;
    size_t actualSrcFileSize = srcFileSize;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input0.bin").c_str(), actualSrcFileSize, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
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