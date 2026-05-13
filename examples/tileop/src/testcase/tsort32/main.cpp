// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tsort32 ST — case-table driven.
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
void LaunchTSORT32_f32_1x32(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_1x64(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_2x32(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_16x32(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_2x64_shared_idx(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_16x64_shared_idx(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_1x8192(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f16_1x32(uint16_t *src, uint32_t *idx, uint16_t *dst, void *stream);
void LaunchTSORT32_f16_4x64(uint16_t *src, uint32_t *idx, uint16_t *dst, void *stream);
void LaunchTSORT32_f32_2x13(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_1x4164(float *src, uint32_t *idx, float *dst, void *stream);
void LaunchTSORT32_f32_2x2084(float *src, uint32_t *idx, float *dst, void *stream);

using LaunchFn = void (*)(void *, uint32_t *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      srcRows;
    size_t      srcCols;
    size_t      idxRows;
    size_t      idxCols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      elemSize;    // bytes per element
};

static const TestCase kCases[] = {
    {"f32_1x32",               reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_1x32),               1,  32,    1,  32,    1,  128,    sizeof(float)},
    {"f32_1x64",               reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_1x64),               1,  64,    1,  64,    1,  256,    sizeof(float)},
    {"f32_2x32",               reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_2x32),               2,  32,    2,  32,    2,  128,    sizeof(float)},
    {"f32_16x32",              reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_16x32),              16, 32,    16, 32,    16, 128,    sizeof(float)},
    {"f32_2x64_shared_idx",    reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_2x64_shared_idx),    2,  64,    1,  64,    2,  256,    sizeof(float)},
    {"f32_16x64_shared_idx",   reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_16x64_shared_idx),   16, 64,    1,  64,    16, 256,    sizeof(float)},
    {"f32_1x8192",             reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_1x8192),             1,  8192,  1,  8192,  1,  32768,  sizeof(float)},
    {"f16_1x32",               reinterpret_cast<LaunchFn>(LaunchTSORT32_f16_1x32),               1,  32,    1,  32,    1,  128,    sizeof(uint16_t)},
    {"f16_4x64",               reinterpret_cast<LaunchFn>(LaunchTSORT32_f16_4x64),               4,  64,    4,  64,    4,  256,    sizeof(uint16_t)},
    {"f32_2x13",               reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_2x13),               2,  16,    2,  16,    2,  64,     sizeof(float)},
    {"f32_1x4164",             reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_1x4164),             1,  8192,  1,  8192,  1,  32768,  sizeof(float)},
    {"f32_2x2084",             reinterpret_cast<LaunchFn>(LaunchTSORT32_f32_2x2084),             2,  3072,  2,  3072,  2,  12288,  sizeof(float)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, aclrtStream stream) {
    int rc = 0;
    size_t srcFileSize = tc.srcRows * tc.srcCols * tc.elemSize;
    size_t idxFileSize = tc.idxRows * tc.idxCols * sizeof(uint32_t);
    size_t dstFileSize = tc.dstRows * tc.dstCols * tc.elemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu, idx=%zux%zu, dst=%zux%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.idxRows, tc.idxCols, tc.dstRows, tc.dstCols);

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr, *dstHost = nullptr;
    uint32_t *idxHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;
    uint32_t *idxDevice = nullptr;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&idxHost), idxFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), srcFileSize, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/idx.bin").c_str(), idxFileSize, idxHost, idxFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/idx.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(idxDevice, idxFileSize, idxHost, idxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, idxDevice, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr)
        aclrtFree(srcDevice);
    if (idxDevice != nullptr)
        aclrtFree(idxDevice);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (srcHost != nullptr)
        aclrtFreeHost(srcHost);
    if (idxHost != nullptr)
        aclrtFreeHost(idxHost);
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
        int ret = RunCase(kCases[i], stream);
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