// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tlrelu ST — case-table driven.
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
#include <fstream>

using namespace PtoTestCommon;

// Kernel launch wrappers (defined in launch.cpp)
void LaunchTLRELU_f32_32x64_dst128(float *src, float *dst, float slope, void *stream);
void LaunchTLRELU_f16_63x64_dst128(uint16_t *src, uint16_t *dst, float slope, void *stream);
void LaunchTLRELU_f32_7x448_dst512(float *src, float *dst, float slope, void *stream);
void LaunchTLRELU_f32_256x16_dst32(float *src, float *dst, float slope, void *stream);

using LaunchFn = void (*)(void *, void *, float, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      srcRows;      // src tile rows
    size_t      srcCols;      // src tile cols
    size_t      dstRows;      // dst tile rows (may have padding)
    size_t      dstCols;      // dst tile cols (may have padding)
    size_t      validRows;    // effective computation rows  (<= srcRows, dstRows)
    size_t      validCols;    // effective computation cols  (<= srcCols, dstCols)
    size_t      elemSize;     // bytes per element
    bool        isFp16;       // true for float16 case
};

static const TestCase kCases[] = {
    {"f32_32x64_dst128",    (LaunchFn)LaunchTLRELU_f32_32x64_dst128,    32,   64,   32,  128, 32,  64,  sizeof(float),  false},
    {"f16_63x64_dst128",    (LaunchFn)LaunchTLRELU_f16_63x64_dst128,    63,   64,   63,  128, 63,  64,  sizeof(uint16_t), true},
    {"f32_7x448_dst512",    (LaunchFn)LaunchTLRELU_f32_7x448_dst512,    7,    448,  7,   512, 7,   448, sizeof(float),  false},
    {"f32_256x16_dst32",    (LaunchFn)LaunchTLRELU_f32_256x16_dst32,    256,  16,   256, 32,  256, 16,  sizeof(float),  false},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t srcFileSize = tc.srcRows * tc.srcCols * tc.elemSize;
    size_t dstFileSize = tc.dstRows * tc.dstCols * tc.elemSize;
    size_t actualSize = 0;

    std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols, tc.validRows, tc.validCols);

    // Per-case data directory
    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;
    float slope = 0.0f;

    aclrtMallocHost(&srcHost, srcFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);

    aclrtMalloc(&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile(caseDir + "/input.bin", actualSize, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    // Read slope (4 bytes float)
    if (rc == 0) {
        std::ifstream slopeFile(caseDir + "/slope.bin", std::ios::binary);
        if (!slopeFile) {
            std::fprintf(stderr, "[ERROR] failed to open %s/slope.bin\n", caseDir.c_str());
            rc = 1;
        } else {
            slopeFile.read(reinterpret_cast<char *>(&slope), sizeof(float));
            slopeFile.close();
        }
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, dstDevice, slope, stream);

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
    // Optional case filter: ./tlrelu [case_name]
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