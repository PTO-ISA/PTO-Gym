// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tcolargmin ST — case-table driven.
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
void LaunchTCOLARGMIN_f32_1x256(int32_t *dst, float *tmp, float *src, void *stream);
void LaunchTCOLARGMIN_f32_16x128(int32_t *dst, float *tmp, float *src, void *stream);
void LaunchTCOLARGMIN_f32_16x256(int32_t *dst, float *tmp, float *src, void *stream);
void LaunchTCOLARGMIN_f16_1x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_f16_16x128(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_f16_16x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui32_1x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui32_16x128(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui32_16x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui16_1x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui16_16x128(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui16_16x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui8_1x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui8_16x128(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_ui8_16x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_i8_1x256(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_i8_16x128(void *dst, void *tmp, void *src, void *stream);
void LaunchTCOLARGMIN_i8_16x256(void *dst, void *tmp, void *src, void *stream);

using LaunchFnFloat = void (*)(int32_t *, float *, float *, void *);
using LaunchFnVoid = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    void *launch;
    size_t      srcRows;
    size_t      srcCols;
    size_t      srcValidRows;
    size_t      srcValidCols;
    size_t      tmpRows;
    size_t      tmpCols;
    size_t      tmpValidRows;
    size_t      tmpValidCols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      dstValidCols;
    size_t      srcElemSize;
    size_t      dstElemSize;
    bool        isFp16;
    bool        isUi32;
    bool        isUi16;
    bool        isUi8;
    bool        isI8;
};

static const TestCase kCases[] = {
    {"f32_1x256", (void*)LaunchTCOLARGMIN_f32_1x256, 1, 256, 1, 255, 1, 256, 1, 255, 1, 256, 255, sizeof(float), sizeof(int32_t), false, false, false, false, false},
    {"f32_16x128", (void*)LaunchTCOLARGMIN_f32_16x128, 16, 128, 16, 127, 16, 128, 16, 127, 1, 128, 127, sizeof(float), sizeof(int32_t), false, false, false, false, false},
    {"f32_16x256", (void*)LaunchTCOLARGMIN_f32_16x256, 16, 256, 15, 255, 16, 256, 15, 255, 1, 256, 255, sizeof(float), sizeof(int32_t), false, false, false, false, false},
    {"f16_1x256", (void*)LaunchTCOLARGMIN_f16_1x256, 1, 256, 1, 255, 1, 256, 1, 255, 1, 256, 255, 2, sizeof(int32_t), true, false, false, false, false},
    {"f16_16x128", (void*)LaunchTCOLARGMIN_f16_16x128, 16, 128, 16, 127, 16, 128, 16, 127, 1, 128, 127, 2, sizeof(int32_t), true, false, false, false, false},
    {"f16_16x256", (void*)LaunchTCOLARGMIN_f16_16x256, 16, 256, 15, 255, 16, 256, 15, 255, 1, 256, 255, 2, sizeof(int32_t), true, false, false, false, false},
    {"ui32_1x256", (void*)LaunchTCOLARGMIN_ui32_1x256, 1, 256, 1, 255, 1, 256, 1, 255, 1, 256, 255, sizeof(uint32_t), sizeof(int32_t), false, true, false, false, false},
    {"ui32_16x128", (void*)LaunchTCOLARGMIN_ui32_16x128, 16, 128, 16, 127, 16, 128, 16, 127, 1, 128, 127, sizeof(uint32_t), sizeof(int32_t), false, true, false, false, false},
    {"ui32_16x256", (void*)LaunchTCOLARGMIN_ui32_16x256, 16, 256, 15, 255, 16, 256, 15, 255, 1, 256, 255, sizeof(uint32_t), sizeof(int32_t), false, true, false, false, false},
    {"ui16_1x256", (void*)LaunchTCOLARGMIN_ui16_1x256, 1, 256, 1, 255, 1, 256, 1, 255, 1, 256, 255, sizeof(uint16_t), sizeof(int32_t), false, false, true, false, false},
    {"ui16_16x128", (void*)LaunchTCOLARGMIN_ui16_16x128, 16, 128, 16, 127, 16, 128, 16, 127, 1, 128, 127, sizeof(uint16_t), sizeof(int32_t), false, false, true, false, false},
    {"ui16_16x256", (void*)LaunchTCOLARGMIN_ui16_16x256, 16, 256, 15, 255, 16, 256, 15, 255, 1, 256, 255, sizeof(uint16_t), sizeof(int32_t), false, false, true, false, false},
    {"ui8_1x256", (void*)LaunchTCOLARGMIN_ui8_1x256, 1, 256, 1, 255, 1, 256, 1, 255, 1, 256, 255, sizeof(uint8_t), sizeof(int32_t), false, false, false, true, false},
    {"ui8_16x128", (void*)LaunchTCOLARGMIN_ui8_16x128, 16, 128, 16, 127, 16, 128, 16, 127, 1, 128, 127, sizeof(uint8_t), sizeof(int32_t), false, false, false, true, false},
    {"ui8_16x256", (void*)LaunchTCOLARGMIN_ui8_16x256, 16, 256, 15, 255, 16, 256, 15, 255, 1, 256, 255, sizeof(uint8_t), sizeof(int32_t), false, false, false, true, false},
    {"i8_1x256", (void*)LaunchTCOLARGMIN_i8_1x256, 1, 256, 1, 255, 1, 256, 1, 255, 1, 256, 255, sizeof(int8_t), sizeof(int32_t), false, false, false, false, true},
    {"i8_16x128", (void*)LaunchTCOLARGMIN_i8_16x128, 16, 128, 16, 127, 16, 128, 16, 127, 1, 128, 127, sizeof(int8_t), sizeof(int32_t), false, false, false, false, true},
    {"i8_16x256", (void*)LaunchTCOLARGMIN_i8_16x256, 16, 256, 15, 255, 16, 256, 15, 255, 1, 256, 255, sizeof(int8_t), sizeof(int32_t), false, false, false, false, true},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcElemCount = tc.srcRows * tc.srcCols;
    const size_t srcFileSize  = srcElemCount * tc.srcElemSize;
    const size_t tmpElemCount = tc.tmpRows * tc.tmpCols;
    const size_t tmpFileSize  = tmpElemCount * tc.srcElemSize;
    const size_t dstElemCount = tc.dstRows * tc.dstCols;
    const size_t dstFileSize  = dstElemCount * tc.dstElemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu, tmp=%zux%zu, dst=%zux%zu, fp16=%d) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.tmpRows, tc.tmpCols, tc.dstRows, tc.dstCols, tc.isFp16);

    std::string caseDir = std::string("./") + tc.name;
    size_t srcFileSizeVar = srcFileSize;
    size_t tmpFileSizeVar = tmpFileSize;
    size_t dstFileSizeVar = dstFileSize;

    void *srcHost = nullptr, *tmpHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *tmpDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&srcHost, srcFileSize);
    aclrtMallocHost(&tmpHost, tmpFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);

    aclrtMalloc(&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&tmpDevice, tmpFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), srcFileSizeVar, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        if (tc.isFp16 || tc.isUi32 || tc.isUi16 || tc.isUi8 || tc.isI8) {
            LaunchFnVoid launch = (LaunchFnVoid)tc.launch;
            launch(dstDevice, tmpDevice, srcDevice, stream);
        } else {
            LaunchFnFloat launch = (LaunchFnFloat)tc.launch;
            launch((int32_t*)dstDevice, (float*)tmpDevice, (float*)srcDevice, stream);
        }

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSizeVar)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr)
        aclrtFree(srcDevice);
    if (tmpDevice != nullptr)
        aclrtFree(tmpDevice);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (srcHost != nullptr)
        aclrtFreeHost(srcHost);
    if (tmpHost != nullptr)
        aclrtFreeHost(tmpHost);
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