// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tpartmin ST — case-table driven.
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
void LaunchTPARTMIN_f32_64x64_full(float *a, float *b, float *c, void *stream);
void LaunchTPARTMIN_f32_2x24_src1_col_less(float *a, float *b, float *c, void *stream);
void LaunchTPARTMIN_f32_128x64_src1_row_less(float *a, float *b, float *c, void *stream);
void LaunchTPARTMIN_f32_95x95_full(float *a, float *b, float *c, void *stream);
void LaunchTPARTMIN_f32_122x123_complex(float *a, float *b, float *c, void *stream);
void LaunchTPARTMIN_f16_122x123_complex(uint16_t *a, uint16_t *b, uint16_t *c, void *stream);
void LaunchTPARTMIN_i16_122x123_complex(int16_t *a, int16_t *b, int16_t *c, void *stream);
void LaunchTPARTMIN_i32_122x123_complex(int32_t *a, int32_t *b, int32_t *c, void *stream);
void LaunchTPARTMIN_u16_122x123_complex(uint16_t *a, uint16_t *b, uint16_t *c, void *stream);
void LaunchTPARTMIN_u32_122x123_complex(uint32_t *a, uint32_t *b, uint32_t *c, void *stream);
void LaunchTPARTMIN_i8_122x123_complex(int8_t *a, int8_t *b, int8_t *c, void *stream);
void LaunchTPARTMIN_u8_122x123_complex(uint8_t *a, uint8_t *b, uint8_t *c, void *stream);

using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      rows;        // allocated tile rows
    size_t      cols;        // allocated tile cols (valid cols)
    size_t      src0ValidRows;  // src0 effective rows
    size_t      src0ValidCols;  // src0 effective cols
    size_t      src1ValidRows;  // src1 effective rows
    size_t      src1ValidCols;  // src1 effective cols
    size_t      dstValidRows;   // dst effective rows
    size_t      dstValidCols;   // dst effective cols
    size_t      elemSize;    // bytes per element
};

static const TestCase kCases[] = {
    {"f32_64x64_full",           reinterpret_cast<LaunchFn>(LaunchTPARTMIN_f32_64x64_full),           64, 64, 64, 64, 64, 64, 64, 64, sizeof(float)},
    {"f32_2x24_src1_col_less",   reinterpret_cast<LaunchFn>(LaunchTPARTMIN_f32_2x24_src1_col_less),    2, 24,  2, 24,  2,  8,  2, 24, sizeof(float)},
    {"f32_128x64_src1_row_less", reinterpret_cast<LaunchFn>(LaunchTPARTMIN_f32_128x64_src1_row_less), 128, 64,128, 64, 96, 64,128, 64, sizeof(float)},
    {"f32_95x95_full",           reinterpret_cast<LaunchFn>(LaunchTPARTMIN_f32_95x95_full),           95, 95, 95, 95, 95, 95, 95, 95, sizeof(float)},
    {"f32_122x123_complex",      reinterpret_cast<LaunchFn>(LaunchTPARTMIN_f32_122x123_complex),      122,123,104,123,122,110,122,123, sizeof(float)},
    {"f16_122x123_complex",      reinterpret_cast<LaunchFn>(LaunchTPARTMIN_f16_122x123_complex),      122,123,104,123,122,110,122,123, sizeof(uint16_t)},
    {"i16_122x123_complex",      reinterpret_cast<LaunchFn>(LaunchTPARTMIN_i16_122x123_complex),      122,123,104,123,122,110,122,123, sizeof(int16_t)},
    {"i32_122x123_complex",      reinterpret_cast<LaunchFn>(LaunchTPARTMIN_i32_122x123_complex),      122,123,104,123,122,110,122,123, sizeof(int32_t)},
    {"u16_122x123_complex",      reinterpret_cast<LaunchFn>(LaunchTPARTMIN_u16_122x123_complex),      122,123,104,123,122,110,122,123, sizeof(uint16_t)},
    {"u32_122x123_complex",      reinterpret_cast<LaunchFn>(LaunchTPARTMIN_u32_122x123_complex),      122,123,104,123,122,110,122,123, sizeof(uint32_t)},
    {"i8_122x123_complex",       reinterpret_cast<LaunchFn>(LaunchTPARTMIN_i8_122x123_complex),       122,123,104,123,122,110,122,123, sizeof(int8_t)},
    {"u8_122x123_complex",       reinterpret_cast<LaunchFn>(LaunchTPARTMIN_u8_122x123_complex),       122,123,104,123,122,110,122,123, sizeof(uint8_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

// Calculate aligned cols for 32-byte alignment
static size_t CalcAlignedCols(size_t cols, size_t elemSize) {
    size_t totalBytes = cols * elemSize;
    size_t alignedBytes = ((totalBytes + 31) / 32) * 32;
    return alignedBytes / elemSize;
}

// Helper to pad data with stride
static void PadDataWithStride(const void *src, void *dst, size_t rows, size_t cols,
                              size_t alignedCols, size_t elemSize) {
    const char *srcPtr = static_cast<const char *>(src);
    char *dstPtr = static_cast<char *>(dst);
    for (size_t r = 0; r < rows; ++r) {
        memcpy(dstPtr + r * alignedCols * elemSize,
               srcPtr + r * cols * elemSize,
               cols * elemSize);
        // Zero-fill padding region (optional, data will be overwritten by kernel)
        memset(dstPtr + r * alignedCols * elemSize + cols * elemSize,
               0,
               (alignedCols - cols) * elemSize);
    }
}

// Helper to unpad data (extract valid cols)
static void UnpadDataWithStride(const void *src, void *dst, size_t rows, size_t cols,
                                size_t alignedCols, size_t elemSize) {
    const char *srcPtr = static_cast<const char *>(src);
    char *dstPtr = static_cast<char *>(dst);
    for (size_t r = 0; r < rows; ++r) {
        memcpy(dstPtr + r * cols * elemSize,
               srcPtr + r * alignedCols * elemSize,
               cols * elemSize);
    }
}

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t fileSize  = elemCount * tc.elemSize;
    const size_t alignedCols = CalcAlignedCols(tc.cols, tc.elemSize);
    const size_t paddedSize = tc.rows * alignedCols * tc.elemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, src0_valid=%zux%zu, src1_valid=%zux%zu, dst_valid=%zux%zu, alignedCols=%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.src0ValidRows, tc.src0ValidCols,
                tc.src1ValidRows, tc.src1ValidCols, tc.dstValidRows, tc.dstValidCols, alignedCols);

    // Per-case data directory
    std::string caseDir = std::string("./") + tc.name;

    void *src0HostOrig = nullptr, *src1HostOrig = nullptr, *dstHostOrig = nullptr;
    void *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

    // Allocate host buffers for original data (contiguous)
    aclrtMallocHost((void **)(&src0HostOrig), fileSize);
    aclrtMallocHost((void **)(&src1HostOrig), fileSize);
    aclrtMallocHost((void **)(&dstHostOrig), fileSize);

    // Allocate host buffers for padded data
    aclrtMallocHost((void **)(&src0Host), paddedSize);
    aclrtMallocHost((void **)(&src1Host), paddedSize);
    aclrtMallocHost((void **)(&dstHost), paddedSize);

    // Allocate device buffers with padded size
    aclrtMalloc((void **)&src0Device, paddedSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, paddedSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, paddedSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (rc == 0) {
        size_t src0FileSize = fileSize;
        size_t src1FileSize = fileSize;
        if (!ReadFile((caseDir + "/input1.bin").c_str(), src0FileSize, src0HostOrig, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), src1FileSize, src1HostOrig, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
            rc = 1;
        }
    }

    if (rc == 0) {
        // Pad input data with stride
        PadDataWithStride(src0HostOrig, src0Host, tc.rows, tc.cols, alignedCols, tc.elemSize);
        PadDataWithStride(src1HostOrig, src1Host, tc.rows, tc.cols, alignedCols, tc.elemSize);

        aclrtMemcpy(src0Device, paddedSize, src0Host, paddedSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, paddedSize, src1Host, paddedSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(src0Device, src1Device, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, paddedSize, dstDevice, paddedSize, ACL_MEMCPY_DEVICE_TO_HOST);

        // Unpad output data
        UnpadDataWithStride(dstHost, dstHostOrig, tc.rows, tc.cols, alignedCols, tc.elemSize);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHostOrig, fileSize)) {
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
    if (src0HostOrig != nullptr)
        aclrtFreeHost(src0HostOrig);
    if (src1HostOrig != nullptr)
        aclrtFreeHost(src1HostOrig);
    if (dstHostOrig != nullptr)
        aclrtFreeHost(dstHostOrig);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    // Optional case filter: ./tpartmin [case_name]
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