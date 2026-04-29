// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tload/tstore ST.
// Each case performs a GM -> Tile -> GM round trip and compare.py checks that
// output.bin matches input.bin exactly for the requested layout.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

void LaunchTLOAD_ND_f32_16x64(float *src, float *dst, void *stream);
void LaunchTLOAD_DN_f32_16x64(float *src, float *dst, void *stream);
void LaunchTLOAD_NZ_f32_128x128(float *src, float *dst, void *stream);
void LaunchTLOAD_ND_PAD_ZERO_f32_16x64(float *src, float *dst, void *stream);
void LaunchTLOAD_DN_PAD_MAX_f32_16x64(float *src, float *dst, void *stream);
void LaunchTLOAD_NZ_PAD_MIN_f32_128x128(float *src, float *dst, void *stream);

using LaunchFn = void (*)(float *, float *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      rows;
    size_t      cols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"nd_f32_16x64",    LaunchTLOAD_ND_f32_16x64,    16, 64,  sizeof(float)},
    {"dn_f32_16x64",    LaunchTLOAD_DN_f32_16x64,    16, 64,  sizeof(float)},
    {"nz_f32_128x128",  LaunchTLOAD_NZ_f32_128x128,  128, 128, sizeof(float)},
    {"nd_pad_zero_f32_16x64", LaunchTLOAD_ND_PAD_ZERO_f32_16x64, 16, 64, sizeof(float)},
    {"dn_pad_max_f32_16x64", LaunchTLOAD_DN_PAD_MAX_f32_16x64, 16, 64, sizeof(float)},
    {"nz_pad_min_f32_128x128", LaunchTLOAD_NZ_PAD_MIN_f32_128x128, 128, 128, sizeof(float)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t fileSize = elemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (%zux%zu) ===\n", tc.name, tc.rows, tc.cols);

    std::string caseDir = std::string("./") + tc.name;
    size_t inputFileSize = fileSize;

    float *srcHost = nullptr;
    float *dstHost = nullptr;
    float *srcDevice = nullptr;
    float *dstDevice = nullptr;

    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), inputFileSize, srcHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        tc.launch(srcDevice, dstDevice, stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSize)) {
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
    bool matchedCase = (caseFilter == nullptr);
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
        matchedCase = true;
        int ret = RunCase(kCases[i], stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (!matchedCase) {
        std::fprintf(stderr, "[ERROR] unknown case filter: %s\n", caseFilter);
        std::fprintf(stderr, "[ERROR] supported cases:");
        for (size_t i = 0; i < kNumCases; ++i) {
            std::fprintf(stderr, " %s", kCases[i].name);
        }
        std::fprintf(stderr, "\n");
        rc = 1;
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}
