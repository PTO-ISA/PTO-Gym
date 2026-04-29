// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trowsum ST — aligned with pto-isa 20 cases.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

// f32 launch wrappers
void LaunchTROWSUM_case1(float *src, float *dst, void *stream);
void LaunchTROWSUM_case2(float *src, float *dst, void *stream);
void LaunchTROWSUM_case3(float *src, float *dst, void *stream);
void LaunchTROWSUM_case4(float *src, float *dst, void *stream);
void LaunchTROWSUM_case5(float *src, float *dst, void *stream);
void LaunchTROWSUM_case6(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTROWSUM_case7(float *src, float *dst, void *stream);
void LaunchTROWSUM_case8(float *src, float *dst, void *stream);
void LaunchTROWSUM_case9(float *src, float *dst, void *stream);
void LaunchTROWSUM_case10(float *src, float *dst, void *stream);

// i32 launch wrappers
void LaunchTROWSUM_case11(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_case12(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_case13(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_case14(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_case15(int32_t *src, int32_t *dst, void *stream);

// i16 launch wrappers
void LaunchTROWSUM_case16(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_case17(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_case18(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_case19(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_case20(int16_t *src, int16_t *dst, void *stream);

using LaunchFnF32 = void (*)(float *, float *, void *);
using LaunchFnF16 = void (*)(uint16_t *, uint16_t *, void *);
using LaunchFnI32 = void (*)(int32_t *, int32_t *, void *);
using LaunchFnI16 = void (*)(int16_t *, int16_t *, void *);

enum class DType { F32, F16, I32, I16 };

struct TestCase {
    const char *name;
    DType       dtype;
    union {
        LaunchFnF32 launchF32;
        LaunchFnF16 launchF16;
        LaunchFnI32 launchI32;
        LaunchFnI16 launchI16;
    };
    size_t      rows;
    size_t      cols;
    size_t      validRows;
    size_t      validCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    // f32 cases
    {"case1",  DType::F32, .launchF32 = LaunchTROWSUM_case1,  127, 64,   127, 63,   4},
    {"case2",  DType::F32, .launchF32 = LaunchTROWSUM_case2,  63,  64,   63,  64,   4},
    {"case3",  DType::F32, .launchF32 = LaunchTROWSUM_case3,  31,  128,  31,  127,  4},
    {"case4",  DType::F32, .launchF32 = LaunchTROWSUM_case4,  15,  192,  15,  192,  4},
    {"case5",  DType::F32, .launchF32 = LaunchTROWSUM_case5,  7,   448,  7,   447,  4},
    {"case6",  DType::F16, .launchF16 = LaunchTROWSUM_case6,  256, 16,   256, 15,   2},
    {"case7",  DType::F32, .launchF32 = LaunchTROWSUM_case7,  64,  128,  64,  128,  4},
    {"case8",  DType::F32, .launchF32 = LaunchTROWSUM_case8,  32,  256,  32,  256,  4},
    {"case9",  DType::F32, .launchF32 = LaunchTROWSUM_case9,  16,  512,  16,  512,  4},
    {"case10", DType::F32, .launchF32 = LaunchTROWSUM_case10, 8,   1024, 8,   1024, 4},

    // i32 cases
    {"case11", DType::I32, .launchI32 = LaunchTROWSUM_case11, 127, 64,   127, 63,   4},
    {"case12", DType::I32, .launchI32 = LaunchTROWSUM_case12, 63,  64,   63,  64,   4},
    {"case13", DType::I32, .launchI32 = LaunchTROWSUM_case13, 31,  128,  31,  127,  4},
    {"case14", DType::I32, .launchI32 = LaunchTROWSUM_case14, 15,  192,  15,  192,  4},
    {"case15", DType::I32, .launchI32 = LaunchTROWSUM_case15, 7,   448,  7,   447,  4},

    // i16 cases
    {"case16", DType::I16, .launchI16 = LaunchTROWSUM_case16, 128, 64,   128, 64,   2},
    {"case17", DType::I16, .launchI16 = LaunchTROWSUM_case17, 64,  64,   64,  64,   2},
    {"case18", DType::I16, .launchI16 = LaunchTROWSUM_case18, 32,  128,  32,  128,  2},
    {"case19", DType::I16, .launchI16 = LaunchTROWSUM_case19, 16,  192,  16,  192,  2},
    {"case20", DType::I16, .launchI16 = LaunchTROWSUM_case20, 8,   448,  8,   448,  2},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcElemCount = tc.rows * tc.cols;
    const size_t srcFileSize  = srcElemCount * tc.elemSize;
    const size_t dstElemCount = tc.rows;
    const size_t dstFileSize  = dstElemCount * tc.elemSize;
    size_t actualFileSize = 0;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;

    void *src0Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&src0Host, srcFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);

    aclrtMalloc(&src0Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), actualFileSize, src0Host, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(src0Device, srcFileSize, src0Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        if (tc.dtype == DType::F32) {
            tc.launchF32((float *)src0Device, (float *)dstDevice, stream);
        } else if (tc.dtype == DType::F16) {
            tc.launchF16((uint16_t *)src0Device, (uint16_t *)dstDevice, stream);
        } else if (tc.dtype == DType::I32) {
            tc.launchI32((int32_t *)src0Device, (int32_t *)dstDevice, stream);
        } else {
            tc.launchI16((int16_t *)src0Device, (int16_t *)dstDevice, stream);
        }

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (src0Device != nullptr)
        aclrtFree(src0Device);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (src0Host != nullptr)
        aclrtFreeHost(src0Host);
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
