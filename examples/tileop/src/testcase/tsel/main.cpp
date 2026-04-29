// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tsel ST — case-table driven.
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
void LaunchTSEL_f32_2x128(uint8_t *mask, float *src0, float *src1, float *dst, void *stream);
void LaunchTSEL_f32_2x32(uint8_t *mask, float *src0, float *src1, float *dst, void *stream);
void LaunchTSEL_f32_2x160(uint8_t *mask, float *src0, float *src1, float *dst, void *stream);
void LaunchTSEL_f32_2x512(uint8_t *mask, float *src0, float *src1, float *dst, void *stream);
void LaunchTSEL_f16_2x128(uint8_t *mask, uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTSEL_f16_2x32(uint8_t *mask, uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTSEL_f16_2x160(uint8_t *mask, uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTSEL_i8_2x128(uint8_t *mask, int8_t *src0, int8_t *src1, int8_t *dst, void *stream);
void LaunchTSEL_i8_2x32(uint8_t *mask, int8_t *src0, int8_t *src1, int8_t *dst, void *stream);
void LaunchTSEL_i8_2x160(uint8_t *mask, int8_t *src0, int8_t *src1, int8_t *dst, void *stream);

enum DataType { DT_F32, DT_F16, DT_I8 };

using LaunchFnF32 = void (*)(uint8_t *, float *, float *, float *, void *);
using LaunchFnF16 = void (*)(uint8_t *, uint16_t *, uint16_t *, uint16_t *, void *);
using LaunchFnI8 = void (*)(uint8_t *, int8_t *, int8_t *, int8_t *, void *);

struct TestCase {
    const char *name;
    DataType    dtype;
    LaunchFnF32 launchF32;
    LaunchFnF16 launchF16;
    LaunchFnI8  launchI8;
    size_t      rows;
    size_t      cols;
    size_t      validRows;
    size_t      validCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"f32_2x128",  DT_F32, LaunchTSEL_f32_2x128,  nullptr,              nullptr,            2, 128,  2, 128,  sizeof(float)},
    {"f32_2x32",   DT_F32, LaunchTSEL_f32_2x32,   nullptr,              nullptr,            2, 32,   2, 32,   sizeof(float)},
    {"f32_2x160",  DT_F32, LaunchTSEL_f32_2x160,  nullptr,              nullptr,            2, 160,  2, 160,  sizeof(float)},
    {"f32_2x512",  DT_F32, LaunchTSEL_f32_2x512,  nullptr,              nullptr,            2, 512,  2, 512,  sizeof(float)},
    {"f16_2x128",  DT_F16, nullptr,              LaunchTSEL_f16_2x128,  nullptr,            2, 128,  2, 128,  sizeof(uint16_t)},
    {"f16_2x32",   DT_F16, nullptr,              LaunchTSEL_f16_2x32,   nullptr,            2, 32,   2, 32,   sizeof(uint16_t)},
    {"f16_2x160",  DT_F16, nullptr,              LaunchTSEL_f16_2x160,  nullptr,            2, 160,  2, 160,  sizeof(uint16_t)},
    {"i8_2x128",   DT_I8,  nullptr,              nullptr,              LaunchTSEL_i8_2x128,  2, 128,  2, 128,  sizeof(int8_t)},
    {"i8_2x32",    DT_I8,  nullptr,              nullptr,              LaunchTSEL_i8_2x32,   2, 32,   2, 32,   sizeof(int8_t)},
    {"i8_2x160",   DT_I8,  nullptr,              nullptr,              LaunchTSEL_i8_2x160,  2, 160,  2, 160,  sizeof(int8_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t fileSizeConst = elemCount * tc.elemSize;
    const size_t maskCols = (tc.validCols + 7) / 8;
    const size_t maskFileSizeConst = tc.validRows * maskCols * sizeof(uint8_t);

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;

    if (tc.dtype == DT_F32) {
        uint8_t *maskHost = nullptr;
        float *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
        uint8_t *maskDevice = nullptr;
        float *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&maskHost), maskFileSizeConst);
        aclrtMallocHost((void **)(&src0Host), fileSizeConst);
        aclrtMallocHost((void **)(&src1Host), fileSizeConst);
        aclrtMallocHost((void **)(&dstHost), fileSizeConst);

        aclrtMalloc((void **)&maskDevice, maskFileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src0Device, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);

        size_t fileSize = fileSizeConst;
        if (!ReadFile((caseDir + "/input1.bin").c_str(), fileSize, src0Host, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }
        fileSize = fileSizeConst;
        if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), fileSize, src1Host, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
            rc = 1;
        }
        size_t maskFileSize = maskFileSizeConst;
        if (rc == 0 && !ReadFile((caseDir + "/input3.bin").c_str(), maskFileSize, maskHost, maskFileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(maskDevice, maskFileSizeConst, maskHost, maskFileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src0Device, fileSizeConst, src0Host, fileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, fileSizeConst, src1Host, fileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);

            tc.launchF32(maskDevice, src0Device, src1Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, fileSizeConst, dstDevice, fileSizeConst, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (maskDevice != nullptr)
            aclrtFree(maskDevice);
        if (src0Device != nullptr)
            aclrtFree(src0Device);
        if (src1Device != nullptr)
            aclrtFree(src1Device);
        if (dstDevice != nullptr)
            aclrtFree(dstDevice);
        if (maskHost != nullptr)
            aclrtFreeHost(maskHost);
        if (src0Host != nullptr)
            aclrtFreeHost(src0Host);
        if (src1Host != nullptr)
            aclrtFreeHost(src1Host);
        if (dstHost != nullptr)
            aclrtFreeHost(dstHost);
    } else if (tc.dtype == DT_F16) {
        uint8_t *maskHost = nullptr;
        uint16_t *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
        uint8_t *maskDevice = nullptr;
        uint16_t *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&maskHost), maskFileSizeConst);
        aclrtMallocHost((void **)(&src0Host), fileSizeConst);
        aclrtMallocHost((void **)(&src1Host), fileSizeConst);
        aclrtMallocHost((void **)(&dstHost), fileSizeConst);

        aclrtMalloc((void **)&maskDevice, maskFileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src0Device, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);

        size_t fileSize = fileSizeConst;
        if (!ReadFile((caseDir + "/input1.bin").c_str(), fileSize, src0Host, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }
        fileSize = fileSizeConst;
        if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), fileSize, src1Host, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
            rc = 1;
        }
        size_t maskFileSize = maskFileSizeConst;
        if (rc == 0 && !ReadFile((caseDir + "/input3.bin").c_str(), maskFileSize, maskHost, maskFileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(maskDevice, maskFileSizeConst, maskHost, maskFileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src0Device, fileSizeConst, src0Host, fileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, fileSizeConst, src1Host, fileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);

            tc.launchF16(maskDevice, src0Device, src1Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, fileSizeConst, dstDevice, fileSizeConst, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (maskDevice != nullptr)
            aclrtFree(maskDevice);
        if (src0Device != nullptr)
            aclrtFree(src0Device);
        if (src1Device != nullptr)
            aclrtFree(src1Device);
        if (dstDevice != nullptr)
            aclrtFree(dstDevice);
        if (maskHost != nullptr)
            aclrtFreeHost(maskHost);
        if (src0Host != nullptr)
            aclrtFreeHost(src0Host);
        if (src1Host != nullptr)
            aclrtFreeHost(src1Host);
        if (dstHost != nullptr)
            aclrtFreeHost(dstHost);
    } else {
        uint8_t *maskHost = nullptr;
        int8_t *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
        uint8_t *maskDevice = nullptr;
        int8_t *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&maskHost), maskFileSizeConst);
        aclrtMallocHost((void **)(&src0Host), fileSizeConst);
        aclrtMallocHost((void **)(&src1Host), fileSizeConst);
        aclrtMallocHost((void **)(&dstHost), fileSizeConst);

        aclrtMalloc((void **)&maskDevice, maskFileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src0Device, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, fileSizeConst, ACL_MEM_MALLOC_HUGE_FIRST);

        size_t fileSize = fileSizeConst;
        if (!ReadFile((caseDir + "/input1.bin").c_str(), fileSize, src0Host, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }
        fileSize = fileSizeConst;
        if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), fileSize, src1Host, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
            rc = 1;
        }
        size_t maskFileSize = maskFileSizeConst;
        if (rc == 0 && !ReadFile((caseDir + "/input3.bin").c_str(), maskFileSize, maskHost, maskFileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(maskDevice, maskFileSizeConst, maskHost, maskFileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src0Device, fileSizeConst, src0Host, fileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, fileSizeConst, src1Host, fileSizeConst, ACL_MEMCPY_HOST_TO_DEVICE);

            tc.launchI8(maskDevice, src0Device, src1Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, fileSizeConst, dstDevice, fileSizeConst, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSizeConst)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (maskDevice != nullptr)
            aclrtFree(maskDevice);
        if (src0Device != nullptr)
            aclrtFree(src0Device);
        if (src1Device != nullptr)
            aclrtFree(src1Device);
        if (dstDevice != nullptr)
            aclrtFree(dstDevice);
        if (maskHost != nullptr)
            aclrtFreeHost(maskHost);
        if (src0Host != nullptr)
            aclrtFreeHost(src0Host);
        if (src1Host != nullptr)
            aclrtFreeHost(src1Host);
        if (dstHost != nullptr)
            aclrtFreeHost(dstHost);
    }

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