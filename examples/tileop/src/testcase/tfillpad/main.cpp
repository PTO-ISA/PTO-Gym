// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tfillpad ST (non-inplace mode).
// Matches C++ reference test cases: Cases 1, 2, 3, 4, 6, 7, 10, 11, 12, 13
// Output size: dst valid region (dst tile physical shape for full output)

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
void LaunchTFILLPAD_f32_128x128_pad_128x127(float *src, float *dst, void *stream);
void LaunchTFILLPAD_f32_128x160_pad_128x127(float *src, float *dst, void *stream);
void LaunchTFILLPAD_f32_128x160_pad_128x127_v2(float *src, float *dst, void *stream);
void LaunchTFILLPAD_f32_260x16_pad_260x7(float *src, float *dst, void *stream);
void LaunchTFILLPAD_u16_260x32_pad_260x7(uint16_t *src, uint16_t *dst, void *stream);
void LaunchTFILLPAD_s8_260x64_pad_260x7(int8_t *src, int8_t *dst, void *stream);
void LaunchTFILLPAD_s16_260x32_pad_260x7(int16_t *src, int16_t *dst, void *stream);
void LaunchTFILLPAD_s32_260x32_pad_260x7(int32_t *src, int32_t *dst, void *stream);
void LaunchTFILLPAD_f32_128x128_pad_128x64_neg1(float *src, float *dst, void *stream);
void LaunchTFILLPAD_f32_128x160_pad_128x127_neg1(float *src, float *dst, void *stream);

enum class DataType { F32, U16, S8, S16, S32 };

struct TestCase {
    const char *name;
    DataType    dtype;
    void (*launch)(void *, void *, void *);
    size_t      rows;        // dst tile rows (physical)
    size_t      cols;        // dst tile cols (physical)
    size_t      validRows;   // dst valid rows (output rows)
    size_t      validCols;   // dst valid cols (output cols) - CHANGED: now = dst physical cols for full output
    size_t      srcRows;     // src tensor rows (0 means same as rows)
    size_t      srcCols;     // src tensor cols (0 means same as cols)
    size_t      elemSize;
};

template<typename T>
void wrapLaunch(void *src, void *dst, void *stream, void (*fn)(T *, T *, void *)) {
    fn((T *)src, (T *)dst, stream);
}

static const TestCase kCases[] = {
    // Case 1: float, src=128x127, dst=128x128, LoadPad=Max, FillPad=Max
    // Output: 128x128 (full dst tile)
    {"f32_128x128_pad_128x127", DataType::F32,
     [](void *src, void *dst, void *stream) { wrapLaunch<float>(src, dst, stream, LaunchTFILLPAD_f32_128x128_pad_128x127); },
     128, 128, 128, 128, 128, 127, sizeof(float)},  // CHANGED: validCols=128, srcCols=127

    // Case 2: float, src=128x127, dst=128x160, LoadPad=Max, FillPad=Max
    // Output: 128x160 (full dst tile)
    {"f32_128x160_pad_128x127", DataType::F32,
     [](void *src, void *dst, void *stream) { wrapLaunch<float>(src, dst, stream, LaunchTFILLPAD_f32_128x160_pad_128x127); },
     128, 160, 128, 160, 128, 127, sizeof(float)},  // CHANGED: validCols=160, srcCols=127

    // Case 3: float, src=128x127, dst=128x160, LoadPad=Min, FillPad=Max
    // Output: 128x160 (full dst tile)
    {"f32_128x160_pad_128x127_v2", DataType::F32,
     [](void *src, void *dst, void *stream) { wrapLaunch<float>(src, dst, stream, LaunchTFILLPAD_f32_128x160_pad_128x127_v2); },
     128, 160, 128, 160, 128, 127, sizeof(float)},  // CHANGED: validCols=160, srcCols=127

    // Case 4: float, src=260x7, dst=260x16, LoadPad=Min, FillPad=Max
    // Output: 260x16 (full dst tile)
    {"f32_260x16_pad_260x7", DataType::F32,
     [](void *src, void *dst, void *stream) { wrapLaunch<float>(src, dst, stream, LaunchTFILLPAD_f32_260x16_pad_260x7); },
     260, 16, 260, 16, 260, 7, sizeof(float)},  // CHANGED: validCols=16, srcCols=7

    // Case 6: uint16, src=260x7, dst=260x32, LoadPad=Min, FillPad=Max
    // Output: 260x32 (full dst tile)
    {"u16_260x32_pad_260x7", DataType::U16,
     [](void *src, void *dst, void *stream) { wrapLaunch<uint16_t>(src, dst, stream, LaunchTFILLPAD_u16_260x32_pad_260x7); },
     260, 32, 260, 32, 260, 7, sizeof(uint16_t)},

    // Case 7: int8, src=260x7, dst=260x64, LoadPad=Min, FillPad=Max
    // Output: 260x64 (full dst tile)
    {"s8_260x64_pad_260x7", DataType::S8,
     [](void *src, void *dst, void *stream) { wrapLaunch<int8_t>(src, dst, stream, LaunchTFILLPAD_s8_260x64_pad_260x7); },
     260, 64, 260, 64, 260, 7, sizeof(int8_t)},  // CHANGED: validCols=64, srcCols=7

    // Case 10: int16, src=260x7, dst=260x32, LoadPad=Min, FillPad=Min
    // Output: 260x32 (full dst tile)
    {"s16_260x32_pad_260x7", DataType::S16,
     [](void *src, void *dst, void *stream) { wrapLaunch<int16_t>(src, dst, stream, LaunchTFILLPAD_s16_260x32_pad_260x7); },
     260, 32, 260, 32, 260, 7, sizeof(int16_t)},  // CHANGED: validCols=32, srcCols=7

    // Case 11: int32, src=260x7, dst=260x32, LoadPad=Min, FillPad=Min
    // Output: 260x32 (full dst tile)
    {"s32_260x32_pad_260x7", DataType::S32,
     [](void *src, void *dst, void *stream) { wrapLaunch<int32_t>(src, dst, stream, LaunchTFILLPAD_s32_260x32_pad_260x7); },
     260, 32, 260, 32, 260, 7, sizeof(int32_t)},  // CHANGED: validCols=32, srcCols=7

    // Case 12: float, src=128x64, dst=128x128, LoadPad=Null, FillPad=Custom(-1.0f)
    // Output: 128x128 (full dst tile)
    {"f32_128x128_pad_128x64_neg1", DataType::F32,
     [](void *src, void *dst, void *stream) { wrapLaunch<float>(src, dst, stream, LaunchTFILLPAD_f32_128x128_pad_128x64_neg1); },
     128, 128, 128, 128, 128, 64, sizeof(float)},  // correct: validCols=128, srcCols=64

    // Case 13: float, src=128x127, dst=128x160, LoadPad=Custom(-1.0f), FillPad=Custom(-1.0f)
    // Output: 128x160 (full dst tile) - CHANGED from 127 to 160
    {"f32_128x160_pad_128x127_neg1", DataType::F32,
     [](void *src, void *dst, void *stream) { wrapLaunch<float>(src, dst, stream, LaunchTFILLPAD_f32_128x160_pad_128x127_neg1); },
     128, 160, 128, 160, 128, 127, sizeof(float)},  // CHANGED: validCols=160
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    size_t srcRows = (tc.srcRows > 0) ? tc.srcRows : tc.rows;
    size_t srcCols = (tc.srcCols > 0) ? tc.srcCols : tc.cols;
    size_t inputElemCount = srcRows * srcCols;
    size_t outputElemCount = tc.validRows * tc.validCols;
    size_t inputFileSize  = inputElemCount * tc.elemSize;
    size_t outputFileSize = outputElemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu, output=%zux%zu) ===\n",
                tc.name, srcRows, srcCols, tc.rows, tc.cols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr, *dstHost = nullptr;
    void *srcDevice = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&srcHost, inputFileSize);
    aclrtMallocHost(&dstHost, outputFileSize);

    aclrtMalloc(&srcDevice, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), inputFileSize, srcHost, inputFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, inputFileSize, srcHost, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(srcDevice, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, outputFileSize)) {
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