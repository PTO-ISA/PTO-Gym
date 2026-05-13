// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tcmp ST — case-table driven.
// Each case launches a different kernel variant, reads/writes from per-case subdirectory.
// Numerical comparison is done externally by compare.py.
// Aligned with testcase/tcmp test cases.

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
void LaunchTCMP_f16_32x32_eq(uint16_t *a, uint16_t *b, int8_t *c, void *stream);
void LaunchTCMP_f32_8x64_gt(float *a, float *b, int8_t *c, void *stream);
void LaunchTCMP_i32_4x64_ne(int32_t *a, int32_t *b, int8_t *c, void *stream);
void LaunchTCMP_i32_128x128_lt(int32_t *a, int32_t *b, int8_t *c, void *stream);
void LaunchTCMP_i32_64x64_eq(int32_t *a, int32_t *b, int8_t *c, void *stream);
void LaunchTCMP_i32_16x32_eq(int32_t *a, int32_t *b, int8_t *c, void *stream);
void LaunchTCMP_f32_128x128_le(float *a, float *b, int8_t *c, void *stream);
void LaunchTCMP_i32_77x80_eq(int32_t *a, int32_t *b, int8_t *c, void *stream);
void LaunchTCMP_i32_32x32_eq(int32_t *a, int32_t *b, int8_t *c, void *stream);
void LaunchTCMP_i16_32x32_eq(int16_t *a, int16_t *b, int8_t *c, void *stream);
void LaunchTCMP_i16_77x80_le(int16_t *a, int16_t *b, int8_t *c, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(void *, void *, void *, void *);
    size_t      rows;
    size_t      cols;
    size_t      srcElemSize;
    size_t      dstElemSize;
};

static const TestCase kCases[] = {
    // Case 1: f16 32x32 eq (half_32x32_32x32)
    {"f16_32x32_eq", (void (*)(void*, void*, void*, void*))LaunchTCMP_f16_32x32_eq, 32, 32, sizeof(uint16_t), sizeof(int8_t)},
    // Case 2: f32 8x64 gt (float_8x64_8x64)
    {"f32_8x64_gt", (void (*)(void*, void*, void*, void*))LaunchTCMP_f32_8x64_gt, 8, 64, sizeof(float), sizeof(int8_t)},
    // Case 3: i32 4x64 ne (int32_4x64_4x64)
    {"i32_4x64_ne", (void (*)(void*, void*, void*, void*))LaunchTCMP_i32_4x64_ne, 4, 64, sizeof(int32_t), sizeof(int8_t)},
    // Case 4: i32 128x128 lt with valid 64x64 (int32_128x128_64x64)
    {"i32_128x128_lt", (void (*)(void*, void*, void*, void*))LaunchTCMP_i32_128x128_lt, 128, 128, sizeof(int32_t), sizeof(int8_t)},
    // Case 5: i32 64x64 eq with valid 32x32 (int32_64x64_32x32)
    {"i32_64x64_eq", (void (*)(void*, void*, void*, void*))LaunchTCMP_i32_64x64_eq, 64, 64, sizeof(int32_t), sizeof(int8_t)},
    // Case 6: i32 16x32 eq (int32_16x32_16x32)
    {"i32_16x32_eq", (void (*)(void*, void*, void*, void*))LaunchTCMP_i32_16x32_eq, 16, 32, sizeof(int32_t), sizeof(int8_t)},
    // Case 7: f32 128x128 le with valid 64x64 (float_128x128_64x64)
    {"f32_128x128_le", (void (*)(void*, void*, void*, void*))LaunchTCMP_f32_128x128_le, 128, 128, sizeof(float), sizeof(int8_t)},
    // Case 8: i32 77x80 eq with valid 32x32 (int32_77x80_32x32)
    {"i32_77x80_eq", (void (*)(void*, void*, void*, void*))LaunchTCMP_i32_77x80_eq, 77, 80, sizeof(int32_t), sizeof(int8_t)},
    // Case 9: i32 32x32 eq (int32_32x32_32x32)
    {"i32_32x32_eq", (void (*)(void*, void*, void*, void*))LaunchTCMP_i32_32x32_eq, 32, 32, sizeof(int32_t), sizeof(int8_t)},
    // Case 10: i16 32x32 eq with valid 16x32 (int16_32x32_16x32)
    {"i16_32x32_eq", (void (*)(void*, void*, void*, void*))LaunchTCMP_i16_32x32_eq, 32, 32, sizeof(int16_t), sizeof(int8_t)},
    // Case 11: i16 77x80 le with valid 32x32 (int16_77x80_32x32)
    {"i16_77x80_le", (void (*)(void*, void*, void*, void*))LaunchTCMP_i16_77x80_le, 77, 80, sizeof(int16_t), sizeof(int8_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcFileSize = tc.rows * tc.cols * tc.srcElemSize;
    const size_t dstFileSize = tc.rows * tc.cols * tc.dstElemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols);

    std::string caseDir = std::string("./") + tc.name;

    void *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

    aclrtMallocHost((void **)(&src0Host), srcFileSize);
    aclrtMallocHost((void **)(&src1Host), srcFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);

    aclrtMalloc((void **)&src0Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t src0FileSize = srcFileSize;
    size_t src1FileSize = srcFileSize;
    size_t dstFileSizeActual = dstFileSize;

    if (!ReadFile((caseDir + "/input1.bin").c_str(), src0FileSize, src0Host, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), src1FileSize, src1Host, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(src0Device, srcFileSize, src0Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, srcFileSize, src1Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(src0Device, src1Device, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSizeActual)) {
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