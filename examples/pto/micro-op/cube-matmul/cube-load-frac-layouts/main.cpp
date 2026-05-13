// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#ifndef TMRGSORT_HPP
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
#endif

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      const char *_recent = aclGetRecentErrMsg();                                \
      if (_recent != nullptr && _recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

#define FILE_CHECK(expr, path)                                                   \
  do {                                                                           \
    if (!(expr)) {                                                               \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",       \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchCube_load_frac_nd2nz_kernel(__fp16 *src, __fp16 *id, float *out, void *stream);

static bool readExact(const char *path, void *dst, size_t size) {
    size_t inputSize = size;
    return ReadFile(path, inputSize, dst, size) && inputSize == size;
}

static bool writeExact(const char *path, void *src, size_t size) {
    return WriteFile(path, src, size);
}

int main() {
    constexpr size_t kNd2NzCase1LhsElem = 40 * 50;
    constexpr size_t kNd2NzCase1RhsElem = 50 * 60;
    constexpr size_t kNd2NzCase1OutElem = 40 * 60;

    constexpr size_t kNd2NzCase1LhsSize = kNd2NzCase1LhsElem * sizeof(__fp16);
    constexpr size_t kNd2NzCase1RhsSize = kNd2NzCase1RhsElem * sizeof(__fp16);
    constexpr size_t kNd2NzCase1OutSize = kNd2NzCase1OutElem * sizeof(float);

    __fp16 *nd2nzCase1LhsHost = nullptr;
    __fp16 *nd2nzCase1RhsHost = nullptr;
    float *outNd2nzCase1Host = nullptr;

    __fp16 *nd2nzCase1LhsDevice = nullptr;
    __fp16 *nd2nzCase1RhsDevice = nullptr;
    float *outNd2nzCase1Device = nullptr;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(envDevice);
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost((void **)(&nd2nzCase1LhsHost), kNd2NzCase1LhsSize));
    ACL_CHECK(aclrtMallocHost((void **)(&nd2nzCase1RhsHost), kNd2NzCase1RhsSize));
    ACL_CHECK(aclrtMallocHost((void **)(&outNd2nzCase1Host), kNd2NzCase1OutSize));

    ACL_CHECK(aclrtMalloc((void **)&nd2nzCase1LhsDevice, kNd2NzCase1LhsSize,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&nd2nzCase1RhsDevice, kNd2NzCase1RhsSize,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&outNd2nzCase1Device, kNd2NzCase1OutSize,
                          ACL_MEM_MALLOC_HUGE_FIRST));

    FILE_CHECK(readExact("./lhs_nd2nz_case1.bin", nd2nzCase1LhsHost, kNd2NzCase1LhsSize),
               "./lhs_nd2nz_case1.bin");
    FILE_CHECK(readExact("./src_nd2nz_case1.bin", nd2nzCase1RhsHost, kNd2NzCase1RhsSize),
               "./src_nd2nz_case1.bin");
    FILE_CHECK(readExact("./out_nd2nz_case1.bin", outNd2nzCase1Host, kNd2NzCase1OutSize),
               "./out_nd2nz_case1.bin");

    ACL_CHECK(aclrtMemcpy(nd2nzCase1LhsDevice, kNd2NzCase1LhsSize, nd2nzCase1LhsHost,
                          kNd2NzCase1LhsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(nd2nzCase1RhsDevice, kNd2NzCase1RhsSize, nd2nzCase1RhsHost,
                          kNd2NzCase1RhsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(outNd2nzCase1Device, kNd2NzCase1OutSize, outNd2nzCase1Host,
                          kNd2NzCase1OutSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchCube_load_frac_nd2nz_kernel(nd2nzCase1RhsDevice, nd2nzCase1LhsDevice,
                                      outNd2nzCase1Device, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(outNd2nzCase1Host, kNd2NzCase1OutSize, outNd2nzCase1Device,
                          kNd2NzCase1OutSize, ACL_MEMCPY_DEVICE_TO_HOST));

    FILE_CHECK(writeExact("./out_nd2nz_case1.bin", outNd2nzCase1Host, kNd2NzCase1OutSize),
               "./out_nd2nz_case1.bin");

cleanup:
    aclrtFree(nd2nzCase1LhsDevice);
    aclrtFree(nd2nzCase1RhsDevice);
    aclrtFree(outNd2nzCase1Device);
    aclrtFreeHost(nd2nzCase1LhsHost);
    aclrtFreeHost(nd2nzCase1RhsHost);
    aclrtFreeHost(outNd2nzCase1Host);
    if (stream != nullptr)
        aclrtDestroyStream(stream);
    if (deviceSet)
        aclrtResetDevice(deviceId);
    if (aclInited)
        aclFinalize();
    return rc;
}
