// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trowexpandadd ST — row-wise broadcast addition.
// Supports multiple data types: f32, f16, i32, i16

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
// f32
void LaunchTROWEXPANDADD_f32_16x32(float *src0, float *src1, float *dst, void *stream);
void LaunchTROWEXPANDADD_f32_56x128(float *src0, float *src1, float *dst, void *stream);
// f16 (use void* for aclFloat16)
void LaunchTROWEXPANDADD_f16_48x64(void *src0, void *src1, void *dst, void *stream);
void LaunchTROWEXPANDADD_f16_16x128(void *src0, void *src1, void *dst, void *stream);
void LaunchTROWEXPANDADD_f16_32x64(void *src0, void *src1, void *dst, void *stream);
// i32
void LaunchTROWEXPANDADD_i32_16x32(void *src0, void *src1, void *dst, void *stream);
// i16
void LaunchTROWEXPANDADD_i16_16x64(void *src0, void *src1, void *dst, void *stream);

// Note: launchTRowExpandAdd2 with src1Col=8 has different semantics - TBD

// Generic launch function type
using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      src0Rows;
    size_t      src0Cols;
    size_t      src1Rows;
    size_t      src1Cols;       // physical src1 cols = 32/sizeof(dtype)
    size_t      dstRows;
    size_t      dstCols;
    size_t      dstValidCols;   // effective dst cols
    size_t      elemSize;
};

static const TestCase kCases[] = {
    // f32 cases
    {"f32_16x32", (LaunchFn)LaunchTROWEXPANDADD_f32_16x32, 16, 32, 16, 8, 16, 32, 32, sizeof(float)},
    {"f32_56x128", (LaunchFn)LaunchTROWEXPANDADD_f32_56x128, 56, 128, 56, 8, 56, 128, 128, sizeof(float)},
    // Note: f32_24x64_v2 and f32_20x64_v2_noeq have different semantics - TBD
    // f16 cases
    {"f16_48x64", LaunchTROWEXPANDADD_f16_48x64, 48, 64, 48, 16, 48, 64, 64, sizeof(uint16_t)},
    {"f16_16x128", LaunchTROWEXPANDADD_f16_16x128, 16, 128, 16, 16, 16, 128, 128, sizeof(uint16_t)},
    {"f16_32x64_noeq", LaunchTROWEXPANDADD_f16_32x64, 32, 64, 32, 16, 32, 64, 64, sizeof(uint16_t)},
    // i32 cases
    {"i32_16x32", LaunchTROWEXPANDADD_i32_16x32, 16, 32, 16, 8, 16, 32, 32, sizeof(int32_t)},
    // i16 cases
    {"i16_16x64", LaunchTROWEXPANDADD_i16_16x64, 16, 64, 16, 16, 16, 64, 64, sizeof(int16_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t src0FileSize = tc.src0Rows * tc.src0Cols * tc.elemSize;
    size_t src1FileSize = tc.src1Rows * tc.src1Cols * tc.elemSize;
    size_t dstFileSize = tc.dstRows * tc.dstCols * tc.elemSize;

    std::printf("[INFO] === case: %s (src0=%zux%zu, src1=%zux%zu, dst=%zux%zu) ===\n",
                tc.name, tc.src0Rows, tc.src0Cols, tc.src1Rows, tc.src1Cols, tc.dstRows, tc.dstCols);

    std::string caseDir = std::string("./") + tc.name;

    void *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);

    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), src0FileSize, src0Host, src0FileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), src1FileSize, src1Host, src1FileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
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