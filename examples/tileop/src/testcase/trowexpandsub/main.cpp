// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trowexpandsub ST — row-wise broadcast subtraction.
// Supports f32, f16, i32, i16

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

// f32 kernels
void LaunchTROWEXPANDSUB_f32_8x128(float *src0, float *src1, float *dst, void *stream);
void LaunchTROWEXPANDSUB_f32_24x32(float *src0, float *src1, float *dst, void *stream);
void LaunchTROWEXPANDSUB_f32_16x128_noeq(float *src0, float *src1, float *dst, void *stream);
// f16 kernels (use void* for aclFloat16)
void LaunchTROWEXPANDSUB_f16_16x256(void *src0, void *src1, void *dst, void *stream);
void LaunchTROWEXPANDSUB_f16_32x64(void *src0, void *src1, void *dst, void *stream);
// i32 kernels
void LaunchTROWEXPANDSUB_i32_16x32(void *src0, void *src1, void *dst, void *stream);
// i16 kernels
void LaunchTROWEXPANDSUB_i16_16x64(void *src0, void *src1, void *dst, void *stream);

// Note: launchTRowExpandSub2 with src1Col>1 has different semantics - TBD

using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols;
    size_t      dstValidRows, dstValidCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    // f32 cases
    {"f32_8x128", (LaunchFn)LaunchTROWEXPANDSUB_f32_8x128, 8, 128, 8, 8, 8, 128, 8, 128, sizeof(float)},
    {"f32_24x32", (LaunchFn)LaunchTROWEXPANDSUB_f32_24x32, 24, 32, 24, 8, 24, 32, 24, 32, sizeof(float)},
    {"f32_16x128_noeq", (LaunchFn)LaunchTROWEXPANDSUB_f32_16x128_noeq, 16, 128, 16, 8, 16, 128, 16, 128, sizeof(float)},
    // f16 cases
    {"f16_16x256", LaunchTROWEXPANDSUB_f16_16x256, 16, 256, 16, 16, 16, 256, 16, 256, sizeof(uint16_t)},
    {"f16_32x64", LaunchTROWEXPANDSUB_f16_32x64, 32, 64, 32, 16, 32, 64, 32, 64, sizeof(uint16_t)},
    // i32 cases
    {"i32_16x32", LaunchTROWEXPANDSUB_i32_16x32, 16, 32, 16, 8, 16, 32, 16, 32, sizeof(int32_t)},
    // i16 cases
    {"i16_16x64", LaunchTROWEXPANDSUB_i16_16x64, 16, 64, 16, 16, 16, 64, 16, 64, sizeof(int16_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t src0FileSize = tc.src0Rows * tc.src0Cols * tc.elemSize;
    size_t src1FileSize = tc.src1Rows * tc.src1Cols * tc.elemSize;
    const size_t dstFileSize = tc.dstRows * tc.dstCols * tc.elemSize;

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

    if (src0Device) aclrtFree(src0Device);
    if (src1Device) aclrtFree(src1Device);
    if (dstDevice) aclrtFree(dstDevice);
    if (src0Host) aclrtFreeHost(src0Host);
    if (src1Host) aclrtFreeHost(src1Host);
    if (dstHost) aclrtFreeHost(dstHost);

    if (rc == 0) std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;
    int rc = 0, deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(envDevice);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter && std::strcmp(kCases[i].name, caseFilter) != 0) continue;
        if (RunCase(kCases[i], deviceId, stream) != 0) { rc = 1; break; }
    }

    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}