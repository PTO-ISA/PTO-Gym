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
      if (_recent != nullptr && _recent[0] != '\0')                             \
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

void LaunchMad_tf32_kernel(float *a, float *b, float *cTf32, float *cPlain,
                           void *stream);

int main() {
  constexpr size_t kM = 16;
  constexpr size_t kN = 16;
  constexpr size_t kK = 16;
  constexpr size_t aElem = kM * kK;
  constexpr size_t bElem = kK * kN;
  constexpr size_t cElem = kM * kN;

  constexpr size_t aSize = aElem * sizeof(float);
  constexpr size_t bSize = bElem * sizeof(float);
  constexpr size_t cSize = cElem * sizeof(float);

  float *aHost = nullptr;
  float *bHost = nullptr;
  float *cTf32Host = nullptr;
  float *cPlainHost = nullptr;
  float *aDevice = nullptr;
  float *bDevice = nullptr;
  float *cTf32Device = nullptr;
  float *cPlainDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t inputSize = 0;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
  ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
  ACL_CHECK(aclrtMallocHost((void **)(&cTf32Host), cSize));
  ACL_CHECK(aclrtMallocHost((void **)(&cPlainHost), cSize));
  ACL_CHECK(aclrtMalloc((void **)&aDevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&bDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&cTf32Device, cSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&cPlainDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = aSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, aHost, aSize) && inputSize == aSize,
             "./v1.bin");
  inputSize = bSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, bHost, bSize) && inputSize == bSize,
             "./v2.bin");
  inputSize = cSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, cTf32Host, cSize) && inputSize == cSize,
             "./v3.bin");
  inputSize = cSize;
  FILE_CHECK(ReadFile("./v4.bin", inputSize, cPlainHost, cSize) && inputSize == cSize,
             "./v4.bin");

  ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(cTf32Device, cSize, cTf32Host, cSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(cPlainDevice, cSize, cPlainHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchMad_tf32_kernel(aDevice, bDevice, cTf32Device, cPlainDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(cTf32Host, cSize, cTf32Device, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(cPlainHost, cSize, cPlainDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v3.bin", cTf32Host, cSize), "./v3.bin");
  FILE_CHECK(WriteFile("./v4.bin", cPlainHost, cSize), "./v4.bin");

cleanup:
  aclrtFree(aDevice);
  aclrtFree(bDevice);
  aclrtFree(cTf32Device);
  aclrtFree(cPlainDevice);
  aclrtFreeHost(aHost);
  aclrtFreeHost(bHost);
  aclrtFreeHost(cTf32Host);
  aclrtFreeHost(cPlainHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
