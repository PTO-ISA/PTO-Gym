// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trowsum ST — case-table driven.

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
void LaunchTROWSUM_f32_127x64_valid127x63(float *src, float *dst, void *stream);
void LaunchTROWSUM_f32_63x64(float *src, float *dst, void *stream);
void LaunchTROWSUM_f32_31x128_valid31x127(float *src, float *dst, void *stream);
void LaunchTROWSUM_f32_15x192(float *src, float *dst, void *stream);
void LaunchTROWSUM_f32_7x448_valid7x447(float *src, float *dst, void *stream);
void LaunchTROWSUM_f16_256x16_valid256x15(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTROWSUM_f32_64x128(float *src, float *dst, void *stream);
void LaunchTROWSUM_f32_32x256(float *src, float *dst, void *stream);
void LaunchTROWSUM_f32_16x512(float *src, float *dst, void *stream);
void LaunchTROWSUM_f32_8x1024(float *src, float *dst, void *stream);
void LaunchTROWSUM_i32_127x64_valid127x63(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_i32_63x64(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_i32_31x128_valid31x127(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_i32_15x192(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_i32_7x448_valid7x447(int32_t *src, int32_t *dst, void *stream);
void LaunchTROWSUM_i16_128x64(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_i16_64x64(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_i16_32x128(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_i16_16x192(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_i16_8x448(int16_t *src, int16_t *dst, void *stream);
void LaunchTROWSUM_i16_1x64_overflow(int16_t *src, int16_t *dst, void *stream);

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
    size_t      rows;       // allocated tile rows
    size_t      cols;       // allocated tile cols
    size_t      validRows;  // effective computation rows  (<= rows)
    size_t      validCols;  // effective computation cols  (<= cols)
    size_t      elemSize;   // bytes per element
};

static const TestCase kCases[] = {
    // f32 cases
    {"f32_127x64_valid127x63",              DType::F32, .launchF32 = LaunchTROWSUM_f32_127x64_valid127x63,              127,  64,  127,  63,  4},
    {"f32_63x64",                           DType::F32, .launchF32 = LaunchTROWSUM_f32_63x64,                           63,   64,  63,   64,  4},
    {"f32_31x128_valid31x127",              DType::F32, .launchF32 = LaunchTROWSUM_f32_31x128_valid31x127,              31,   128, 31,   127, 4},
    {"f32_15x192",                          DType::F32, .launchF32 = LaunchTROWSUM_f32_15x192,                          15,   192, 15,   192, 4},
    {"f32_7x448_valid7x447",                DType::F32, .launchF32 = LaunchTROWSUM_f32_7x448_valid7x447,                7,    448, 7,    447, 4},
    // f16 case
    {"f16_256x16_valid256x15",              DType::F16, .launchF16 = LaunchTROWSUM_f16_256x16_valid256x15,              256,  16,  256,  15,  2},
    // f32 DN dst cases
    {"f32_64x128",                          DType::F32, .launchF32 = LaunchTROWSUM_f32_64x128,                          64,   128, 64,   128, 4},
    {"f32_32x256",                          DType::F32, .launchF32 = LaunchTROWSUM_f32_32x256,                          32,   256, 32,   256, 4},
    {"f32_16x512",                          DType::F32, .launchF32 = LaunchTROWSUM_f32_16x512,                          16,   512, 16,   512, 4},
    {"f32_8x1024",                          DType::F32, .launchF32 = LaunchTROWSUM_f32_8x1024,                          8,    1024,8,    1024,4},
    // int32 cases
    {"i32_127x64_valid127x63",              DType::I32, .launchI32 = LaunchTROWSUM_i32_127x64_valid127x63,              127,  64,  127,  63,  4},
    {"i32_63x64",                           DType::I32, .launchI32 = LaunchTROWSUM_i32_63x64,                           63,   64,  63,   64,  4},
    {"i32_31x128_valid31x127",              DType::I32, .launchI32 = LaunchTROWSUM_i32_31x128_valid31x127,              31,   128, 31,   127, 4},
    {"i32_15x192",                          DType::I32, .launchI32 = LaunchTROWSUM_i32_15x192,                          15,   192, 15,   192, 4},
    {"i32_7x448_valid7x447",                DType::I32, .launchI32 = LaunchTROWSUM_i32_7x448_valid7x447,                7,    448, 7,    447, 4},
    // int16 cases
    {"i16_128x64",                          DType::I16, .launchI16 = LaunchTROWSUM_i16_128x64,                          128,  64,  128,  64,  2},
    {"i16_64x64",                           DType::I16, .launchI16 = LaunchTROWSUM_i16_64x64,                           64,   64,  64,   64,  2},
    {"i16_32x128",                          DType::I16, .launchI16 = LaunchTROWSUM_i16_32x128,                          32,   128, 32,   128, 2},
    {"i16_16x192",                          DType::I16, .launchI16 = LaunchTROWSUM_i16_16x192,                          16,   192, 16,   192, 2},
    {"i16_8x448",                           DType::I16, .launchI16 = LaunchTROWSUM_i16_8x448,                           8,    448, 8,    448, 2},
    // i16 overflow case
    {"i16_1x64_overflow",                   DType::I16, .launchI16 = LaunchTROWSUM_i16_1x64_overflow,                   1,    64,  1,    64,  2},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcElemCount = tc.rows * tc.cols;
    const size_t srcFileSize  = srcElemCount * tc.elemSize;
    const size_t dstElemCount = tc.validRows * 1;
    const size_t dstFileSize  = dstElemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols);

    // Per-case data directory
    std::string caseDir = std::string("./") + tc.name;
    size_t src0FileSize = srcFileSize;

    void *src0Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&src0Host, srcFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);

    aclrtMalloc(&src0Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), src0FileSize, src0Host, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(src0Device, srcFileSize, src0Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        switch (tc.dtype) {
            case DType::F32: tc.launchF32((float *)src0Device, (float *)dstDevice, stream); break;
            case DType::F16: tc.launchF16((uint16_t *)src0Device, (uint16_t *)dstDevice, stream); break;
            case DType::I32: tc.launchI32((int32_t *)src0Device, (int32_t *)dstDevice, stream); break;
            case DType::I16: tc.launchI16((int16_t *)src0Device, (int16_t *)dstDevice, stream); break;
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
    // Optional case filter: ./trowsum [case_name]
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
