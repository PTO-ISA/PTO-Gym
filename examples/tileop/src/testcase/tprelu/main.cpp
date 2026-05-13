// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tprelu ST — case-table driven.
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
void LaunchTPRELU_f16_64x64(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTPRELU_f16_63x63(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTPRELU_f16_1x16384(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTPRELU_f16_2048x16(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream);
void LaunchTPRELU_f32_64x64(float *src0, float *src1, float *dst, void *stream);
void LaunchTPRELU_f32_63x63(float *src0, float *src1, float *dst, void *stream);
void LaunchTPRELU_f32_1x16384(float *src0, float *src1, float *dst, void *stream);
void LaunchTPRELU_f32_2048x8(float *src0, float *src1, float *dst, void *stream);

enum DataType { F16, F32 };

struct TestCase {
    const char *name;
    DataType    dtype;
    void *      launch;
    size_t      rows;
    size_t      cols;
    size_t      validRows;
    size_t      validCols;
};

static const TestCase kCases[] = {
    {"f16_64x64",       F16, (void*)LaunchTPRELU_f16_64x64,       64,    64,     64,    64},
    {"f16_63x63",       F16, (void*)LaunchTPRELU_f16_63x63,       64,    64,     63,    63},
    {"f16_1x16384",     F16, (void*)LaunchTPRELU_f16_1x16384,     1,     16384,  1,     16384},
    {"f16_2048x16",     F16, (void*)LaunchTPRELU_f16_2048x16,     2048,  16,     2048,  16},
    {"f32_64x64",       F32, (void*)LaunchTPRELU_f32_64x64,       64,    64,     64,    64},
    {"f32_63x63",       F32, (void*)LaunchTPRELU_f32_63x63,       64,    64,     63,    63},
    {"f32_1x16384",     F32, (void*)LaunchTPRELU_f32_1x16384,     1,     16384,  1,     16384},
    {"f32_2048x8",      F32, (void*)LaunchTPRELU_f32_2048x8,      2048,  8,      2048,  8},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

template<typename T>
using LaunchFn = void (*)(T *, T *, T *, void *);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t elemSize = (tc.dtype == F16) ? sizeof(uint16_t) : sizeof(float);
    size_t fileSize  = elemCount * elemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu, dtype=%s) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols,
                (tc.dtype == F16) ? "f16" : "f32");

    std::string caseDir = std::string("./") + tc.name;

    if (tc.dtype == F16) {
        uint16_t *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
        uint16_t *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&src0Host), fileSize);
        aclrtMallocHost((void **)(&src1Host), fileSize);
        aclrtMallocHost((void **)(&dstHost), fileSize);

        aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        if (!ReadFile((caseDir + "/input0.bin").c_str(), fileSize, src0Host, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input1.bin").c_str(), fileSize, src1Host, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

            LaunchFn<uint16_t> launch = (LaunchFn<uint16_t>)tc.launch;
            launch(src0Device, src1Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (src0Device != nullptr) aclrtFree(src0Device);
        if (src1Device != nullptr) aclrtFree(src1Device);
        if (dstDevice != nullptr) aclrtFree(dstDevice);
        if (src0Host != nullptr) aclrtFreeHost(src0Host);
        if (src1Host != nullptr) aclrtFreeHost(src1Host);
        if (dstHost != nullptr) aclrtFreeHost(dstHost);
    } else {
        float *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
        float *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

        aclrtMallocHost((void **)(&src0Host), fileSize);
        aclrtMallocHost((void **)(&src1Host), fileSize);
        aclrtMallocHost((void **)(&dstHost), fileSize);

        aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        if (!ReadFile((caseDir + "/input0.bin").c_str(), fileSize, src0Host, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input0.bin\n", caseDir.c_str());
            rc = 1;
        }
        if (rc == 0 && !ReadFile((caseDir + "/input1.bin").c_str(), fileSize, src1Host, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (rc == 0) {
            aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
            aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

            LaunchFn<float> launch = (LaunchFn<float>)tc.launch;
            launch(src0Device, src1Device, dstDevice, stream);

            aclrtSynchronizeStream(stream);
            aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSize)) {
            std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
            rc = 1;
        }

        if (src0Device != nullptr) aclrtFree(src0Device);
        if (src1Device != nullptr) aclrtFree(src1Device);
        if (dstDevice != nullptr) aclrtFree(dstDevice);
        if (src0Host != nullptr) aclrtFreeHost(src0Host);
        if (src1Host != nullptr) aclrtFreeHost(src1Host);
        if (dstHost != nullptr) aclrtFreeHost(dstHost);
    }

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