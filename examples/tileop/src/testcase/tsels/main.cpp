// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

void LaunchTSELS_uint8_uint8_2x32_2x32_2x32_2x32(uint8_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint8_uint16_2x32_2x16_2x32_2x32(uint16_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint8_uint32_2x32_2x8_2x32_2x32(uint32_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint16_uint8_2x16_2x32_2x16_2x16(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint16_uint16_2x16_2x16_2x16_2x16(uint16_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint16_uint32_2x16_2x8_2x16_2x16(uint32_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint32_uint8_2x8_2x32_2x8_2x8(uint8_t *mask, uint32_t *src, uint32_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint32_uint16_2x8_2x16_2x8_2x8(uint16_t *mask, uint32_t *src, uint32_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint32_uint32_2x8_2x8_2x8_2x8(uint32_t *mask, uint32_t *src, uint32_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f16_uint8_2x16_2x32_2x16_2x16(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f16_uint16_2x16_2x16_2x16_2x16(uint16_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f16_uint32_2x16_2x8_2x16_2x16(uint32_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f32_uint8_2x8_2x32_2x8_2x8(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f32_uint16_2x8_2x16_2x8_2x8(uint16_t *mask, float *src, float *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f32_uint32_2x8_2x8_2x8_2x8(uint32_t *mask, float *src, float *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint8_uint8_2x32_2x64_2x128_2x31(uint8_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint16_uint8_2x32_2x64_2x128_2x31(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f32_uint8_2x32_2x64_2x128_2x31(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_uint8_uint8_32x672_32x96_32x672_32x666(uint8_t *mask, uint8_t *src, uint8_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f16_uint8_32x672_32x96_32x672_32x666(uint8_t *mask, uint16_t *src, uint16_t *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f32_uint8_32x672_32x96_32x672_32x666(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream);
void LaunchTSELS_f32_uint8_1x8192_1x4096_1x8192_1x8192(uint8_t *mask, float *src, float *dst, void *scalar_ptr, void *stream);

struct TestCase {
    const char *name;
    void (*launch)(void*, void*, void*, void*, void*);
    size_t dstRows, dstCols;
    size_t maskRows, maskCols;
    size_t srcRows, srcCols;
    size_t validRows, validCols;
    size_t dstElemSize;
    size_t maskElemSize;
    size_t srcElemSize;
};

static const TestCase kCases[] = {
    {"uint8_uint8_2x32_2x32_2x32_2x32", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint8_uint8_2x32_2x32_2x32_2x32, 2, 32, 2, 32, 2, 32, 2, 32, 1, 1, 1},
    {"uint8_uint16_2x32_2x16_2x32_2x32", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint8_uint16_2x32_2x16_2x32_2x32, 2, 32, 2, 16, 2, 32, 2, 32, 1, 2, 1},
    {"uint8_uint32_2x32_2x8_2x32_2x32", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint8_uint32_2x32_2x8_2x32_2x32, 2, 32, 2, 8, 2, 32, 2, 32, 1, 4, 1},
    {"uint16_uint8_2x16_2x32_2x16_2x16", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint16_uint8_2x16_2x32_2x16_2x16, 2, 16, 2, 32, 2, 16, 2, 16, 2, 1, 2},
    {"uint16_uint16_2x16_2x16_2x16_2x16", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint16_uint16_2x16_2x16_2x16_2x16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 2, 2},
    {"uint16_uint32_2x16_2x8_2x16_2x16", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint16_uint32_2x16_2x8_2x16_2x16, 2, 16, 2, 8, 2, 16, 2, 16, 2, 4, 2},
    {"uint32_uint8_2x8_2x32_2x8_2x8", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint32_uint8_2x8_2x32_2x8_2x8, 2, 8, 2, 32, 2, 8, 2, 8, 4, 1, 4},
    {"uint32_uint16_2x8_2x16_2x8_2x8", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint32_uint16_2x8_2x16_2x8_2x8, 2, 8, 2, 16, 2, 8, 2, 8, 4, 2, 4},
    {"uint32_uint32_2x8_2x8_2x8_2x8", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint32_uint32_2x8_2x8_2x8_2x8, 2, 8, 2, 8, 2, 8, 2, 8, 4, 4, 4},
    {"f16_uint8_2x16_2x32_2x16_2x16", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f16_uint8_2x16_2x32_2x16_2x16, 2, 16, 2, 32, 2, 16, 2, 16, 2, 1, 2},
    {"f16_uint16_2x16_2x16_2x16_2x16", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f16_uint16_2x16_2x16_2x16_2x16, 2, 16, 2, 16, 2, 16, 2, 16, 2, 2, 2},
    {"f16_uint32_2x16_2x8_2x16_2x16", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f16_uint32_2x16_2x8_2x16_2x16, 2, 16, 2, 8, 2, 16, 2, 16, 2, 4, 2},
    {"f32_uint8_2x8_2x32_2x8_2x8", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f32_uint8_2x8_2x32_2x8_2x8, 2, 8, 2, 32, 2, 8, 2, 8, 4, 1, 4},
    {"f32_uint16_2x8_2x16_2x8_2x8", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f32_uint16_2x8_2x16_2x8_2x8, 2, 8, 2, 16, 2, 8, 2, 8, 4, 2, 4},
    {"f32_uint32_2x8_2x8_2x8_2x8", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f32_uint32_2x8_2x8_2x8_2x8, 2, 8, 2, 8, 2, 8, 2, 8, 4, 4, 4},
    {"uint8_uint8_2x32_2x64_2x128_2x31", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint8_uint8_2x32_2x64_2x128_2x31, 2, 32, 2, 64, 2, 128, 2, 31, 1, 1, 1},
    {"uint16_uint8_2x32_2x64_2x128_2x31", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint16_uint8_2x32_2x64_2x128_2x31, 2, 32, 2, 64, 2, 128, 2, 31, 2, 1, 2},
    {"f32_uint8_2x32_2x64_2x128_2x31", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f32_uint8_2x32_2x64_2x128_2x31, 2, 32, 2, 64, 2, 128, 2, 31, 4, 1, 4},
    {"uint8_uint8_32x672_32x96_32x672_32x666", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_uint8_uint8_32x672_32x96_32x672_32x666, 32, 672, 32, 96, 32, 672, 32, 666, 1, 1, 1},
    {"f16_uint8_32x672_32x96_32x672_32x666", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f16_uint8_32x672_32x96_32x672_32x666, 32, 672, 32, 96, 32, 672, 32, 666, 2, 1, 2},
    {"f32_uint8_32x672_32x96_32x672_32x666", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f32_uint8_32x672_32x96_32x672_32x666, 32, 672, 32, 96, 32, 672, 32, 666, 4, 1, 4},
    {"f32_uint8_1x8192_1x4096_1x8192_1x8192", (void(*)(void*,void*,void*,void*,void*))LaunchTSELS_f32_uint8_1x8192_1x4096_1x8192_1x8192, 1, 8192, 1, 4096, 1, 8192, 1, 8192, 4, 1, 4},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t dstFileSize = tc.dstRows * tc.dstCols * tc.dstElemSize;
    size_t maskFileSize = tc.maskRows * tc.maskCols * tc.maskElemSize;
    size_t srcFileSize = tc.srcRows * tc.srcCols * tc.srcElemSize;
    size_t scalarFileSize = tc.dstElemSize;

    std::printf("[INFO] === case: %s (dst=%zux%zu, mask=%zux%zu, src=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.dstRows, tc.dstCols, tc.maskRows, tc.maskCols, tc.srcRows, tc.srcCols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;
    const size_t maskFileSizeBuf = maskFileSize;
    const size_t srcFileSizeBuf = srcFileSize;
    const size_t scalarFileSizeBuf = scalarFileSize;

    void *maskHost = nullptr, *srcHost = nullptr, *dstHost = nullptr, *scalarHost = nullptr;
    void *maskDevice = nullptr, *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&maskHost, maskFileSize);
    aclrtMallocHost(&srcHost, srcFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);
    aclrtMallocHost(&scalarHost, scalarFileSize);

    aclrtMalloc(&maskDevice, maskFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    memset(dstHost, 0, dstFileSize);

    if (!ReadFile(caseDir + "/mask.bin", maskFileSize, maskHost, maskFileSizeBuf)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/mask.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile(caseDir + "/input1.bin", srcFileSize, srcHost, srcFileSizeBuf)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile(caseDir + "/input2.bin", scalarFileSize, scalarHost, scalarFileSizeBuf)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(maskDevice, maskFileSize, maskHost, maskFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(maskDevice, srcDevice, dstDevice, scalarHost, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (maskDevice != nullptr) aclrtFree(maskDevice);
    if (srcDevice != nullptr) aclrtFree(srcDevice);
    if (dstDevice != nullptr) aclrtFree(dstDevice);
    if (maskHost != nullptr) aclrtFreeHost(maskHost);
    if (srcHost != nullptr) aclrtFreeHost(srcHost);
    if (dstHost != nullptr) aclrtFreeHost(dstHost);
    if (scalarHost != nullptr) aclrtFreeHost(scalarHost);

    if (rc == 0) std::printf("[INFO] case %s done\n", tc.name);
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

    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}