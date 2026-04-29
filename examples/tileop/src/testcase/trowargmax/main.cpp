// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trowargmax ST — case-table driven.

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
void LaunchTROWARGMAX_uint32_float_8x1_8x8_8x8(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_1024x1_1024x8_1024x8(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_16x1_13x16_13x13(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_1024x1_1023x24_1023x17(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_8x1_8x64_8x64(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_264x1_260x64_260x64(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_8x1_1x128_1x128(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_64x1_32x128_32x128(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_8x1_3x4096_3x4095(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_8x1_2x16384_2x16381(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_16x1_2x16_2x16(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_16x1_13x16_13x13(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_272x1_260x64_260x64(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_16x1_3x8192_3x8191(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_16x1_1x16384_1x16381(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_16x1_1x32768_1x32761(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_int32_float_16x1_13x16_13x13(float *src, int32_t *dst, void *stream);
void LaunchTROWARGMAX_int32_half_16x1_13x16_13x13(uint16_t *src, int32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_3x8_3x3480_3x3473(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_260x8_260x64_260x64(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_float_1023x8_1023x24_1023x17(float *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_3x16_3x3488_3x3473(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_260x16_260x64_260x64(uint16_t *src, uint32_t *dst, void *stream);
void LaunchTROWARGMAX_uint32_half_1023x16_1023x32_1023x17(uint16_t *src, uint32_t *dst, void *stream);

using LaunchFnF32U32 = void (*)(float *, uint32_t *, void *);
using LaunchFnF16U32 = void (*)(uint16_t *, uint32_t *, void *);
using LaunchFnF32S32 = void (*)(float *, int32_t *, void *);
using LaunchFnF16S32 = void (*)(uint16_t *, int32_t *, void *);

enum class DType { F32U32, F16U32, F32S32, F16S32 };

struct TestCase {
    const char *name;
    DType       dtype;
    union {
        LaunchFnF32U32 launchF32U32;
        LaunchFnF16U32 launchF16U32;
        LaunchFnF32S32 launchF32S32;
        LaunchFnF16S32 launchF16S32;
    };
    size_t      rows;       // allocated tile rows
    size_t      cols;       // allocated tile cols
    size_t      validRows;  // effective computation rows  (<= rows)
    size_t      validCols;  // effective computation cols  (<= cols)
    size_t      srcElemSize;   // bytes per src element
    size_t      dstElemSize;   // bytes per dst element
    size_t      dstCols;       // dst tile cols
};

static const TestCase kCases[] = {
    {"uint32_float_8x1_8x8_8x8",              DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_8x1_8x8_8x8,              8,  8,  8,  8,  4, 4, 1},
    {"uint32_float_1024x1_1024x8_1024x8",     DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_1024x1_1024x8_1024x8,     1024,  8,  1024,  8,  4, 4, 1},
    {"uint32_float_16x1_13x16_13x13",         DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_16x1_13x16_13x13,         13,  16,  13,  13,  4, 4, 1},
    {"uint32_float_1024x1_1023x24_1023x17",   DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_1024x1_1023x24_1023x17,   1023,  24,  1023,  17,  4, 4, 1},
    {"uint32_float_8x1_8x64_8x64",            DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_8x1_8x64_8x64,            8,  64,  8,  64,  4, 4, 1},
    {"uint32_float_264x1_260x64_260x64",      DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_264x1_260x64_260x64,      260,  64,  260,  64,  4, 4, 1},
    {"uint32_float_8x1_1x128_1x128",          DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_8x1_1x128_1x128,          1,  128,  1,  128,  4, 4, 1},
    {"uint32_float_64x1_32x128_32x128",       DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_64x1_32x128_32x128,       32,  128,  32,  128,  4, 4, 1},
    {"uint32_float_8x1_3x4096_3x4095",        DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_8x1_3x4096_3x4095,        3,  4096,  3,  4095,  4, 4, 1},
    {"uint32_float_8x1_2x16384_2x16381",      DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_8x1_2x16384_2x16381,      2,  16384,  2,  16381,  4, 4, 1},
    {"uint32_half_16x1_2x16_2x16",            DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_16x1_2x16_2x16,            2,  16,  2,  16,  2, 4, 1},
    {"uint32_half_16x1_13x16_13x13",          DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_16x1_13x16_13x13,          13,  16,  13,  13,  2, 4, 1},
    {"uint32_half_272x1_260x64_260x64",       DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_272x1_260x64_260x64,       260,  64,  260,  64,  2, 4, 1},
    {"uint32_half_16x1_3x8192_3x8191",        DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_16x1_3x8192_3x8191,        3,  8192,  3,  8191,  2, 4, 1},
    {"uint32_half_16x1_1x16384_1x16381",      DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_16x1_1x16384_1x16381,      1,  16384,  1,  16381,  2, 4, 1},
    {"uint32_half_16x1_1x32768_1x32761",      DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_16x1_1x32768_1x32761,      1,  32768,  1,  32761,  2, 4, 1},
    {"int32_float_16x1_13x16_13x13",          DType::F32S32, .launchF32S32 = LaunchTROWARGMAX_int32_float_16x1_13x16_13x13,          13,  16,  13,  13,  4, 4, 1},
    {"int32_half_16x1_13x16_13x13",           DType::F16S32, .launchF16S32 = LaunchTROWARGMAX_int32_half_16x1_13x16_13x13,           13,  16,  13,  13,  2, 4, 1},
    {"uint32_float_3x8_3x3480_3x3473",        DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_3x8_3x3480_3x3473,        3,  3480,  3,  3473,  4, 4, 8},
    {"uint32_float_260x8_260x64_260x64",      DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_260x8_260x64_260x64,      260,  64,  260,  64,  4, 4, 8},
    {"uint32_float_1023x8_1023x24_1023x17",   DType::F32U32, .launchF32U32 = LaunchTROWARGMAX_uint32_float_1023x8_1023x24_1023x17,   1023,  24,  1023,  17,  4, 4, 8},
    {"uint32_half_3x16_3x3488_3x3473",        DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_3x16_3x3488_3x3473,        3,  3488,  3,  3473,  2, 4, 16},
    {"uint32_half_260x16_260x64_260x64",      DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_260x16_260x64_260x64,      260,  64,  260,  64,  2, 4, 16},
    {"uint32_half_1023x16_1023x32_1023x17",   DType::F16U32, .launchF16U32 = LaunchTROWARGMAX_uint32_half_1023x16_1023x32_1023x17,   1023,  32,  1023,  17,  2, 4, 16},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t srcElemCount = tc.rows * tc.cols;
    const size_t srcFileSize  = srcElemCount * tc.srcElemSize;
    const size_t dstElemCount = tc.validRows * tc.dstCols;
    const size_t dstFileSize  = dstElemCount * tc.dstElemSize;

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

    if (rc == 0) {
        aclrtMemset(dstDevice, dstFileSize, 0, dstFileSize);
    }

    if (!ReadFile((caseDir + "/input1.bin").c_str(), src0FileSize, src0Host, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(src0Device, srcFileSize, src0Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        switch (tc.dtype) {
            case DType::F32U32:
                tc.launchF32U32((float *)src0Device, (uint32_t *)dstDevice, stream);
                break;
            case DType::F16U32:
                tc.launchF16U32((uint16_t *)src0Device, (uint32_t *)dstDevice, stream);
                break;
            case DType::F32S32:
                tc.launchF32S32((float *)src0Device, (int32_t *)dstDevice, stream);
                break;
            case DType::F16S32:
                tc.launchF16S32((uint16_t *)src0Device, (int32_t *)dstDevice, stream);
                break;
        }

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0) {
        mkdir(caseDir.c_str(), 0755);
        if (!WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }
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
    // Optional case filter: ./trowargmax [case_name]
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
