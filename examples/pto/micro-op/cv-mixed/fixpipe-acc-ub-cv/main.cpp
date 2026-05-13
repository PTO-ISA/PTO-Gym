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
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",        \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchFixpipe_acc_ub_cv_kernel(__fp16 *src, __fp16 *id, float *out,
                                    void *stream);

int main() {
  constexpr size_t kSrcElem = 50 * 64;
  constexpr size_t kIdElem = 40 * 50;
  constexpr size_t kOutElem = 40 * 64;
  constexpr size_t kSrcSize = kSrcElem * sizeof(__fp16);
  constexpr size_t kIdSize = kIdElem * sizeof(__fp16);
  constexpr size_t kOutSize = kOutElem * sizeof(float);

  __fp16 *srcHost = nullptr;
  __fp16 *idHost = nullptr;
  float *outHost = nullptr;
  __fp16 *srcDevice = nullptr;
  __fp16 *idDevice = nullptr;
  float *outDevice = nullptr;

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

  ACL_CHECK(aclrtMallocHost((void **)&srcHost, kSrcSize));
  ACL_CHECK(aclrtMallocHost((void **)&idHost, kIdSize));
  ACL_CHECK(aclrtMallocHost((void **)&outHost, kOutSize));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kSrcSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&idDevice, kIdSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kIdSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, idHost, kIdSize) && inputSize == kIdSize,
             "./v1.bin");
  inputSize = kSrcSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, srcHost, kSrcSize) && inputSize == kSrcSize,
             "./v2.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, outHost, kOutSize) && inputSize == kOutSize,
             "./v3.bin");

  ACL_CHECK(aclrtMemcpy(srcDevice, kSrcSize, srcHost, kSrcSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(idDevice, kIdSize, idHost, kIdSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, kOutSize, outHost, kOutSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_acc_ub_cv_kernel(srcDevice, idDevice, outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outHost, kOutSize, outDevice, kOutSize, ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v3.bin", outHost, kOutSize), "./v3.bin");

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(idDevice);
  aclrtFree(outDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(idHost);
  aclrtFreeHost(outHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
