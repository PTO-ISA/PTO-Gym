// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

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

void LaunchMixed_external_vadd_kernel(float *lhs, float *rhs, float *out,
                                      void *stream);

int main() {
  constexpr size_t elemCount = 1024;
  constexpr size_t bufSize = elemCount * sizeof(float);

  float *lhsHost = nullptr;
  float *rhsHost = nullptr;
  float *outHost = nullptr;
  float *lhsDevice = nullptr;
  float *rhsDevice = nullptr;
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

  ACL_CHECK(aclrtMallocHost((void **)&lhsHost, bufSize));
  ACL_CHECK(aclrtMallocHost((void **)&rhsHost, bufSize));
  ACL_CHECK(aclrtMallocHost((void **)&outHost, bufSize));
  ACL_CHECK(aclrtMalloc((void **)&lhsDevice, bufSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&rhsDevice, bufSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, bufSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = bufSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, lhsHost, bufSize) &&
                 inputSize == bufSize,
             "./v1.bin");
  inputSize = bufSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, rhsHost, bufSize) &&
                 inputSize == bufSize,
             "./v2.bin");
  inputSize = bufSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, outHost, bufSize) &&
                 inputSize == bufSize,
             "./v3.bin");

  ACL_CHECK(aclrtMemcpy(lhsDevice, bufSize, lhsHost, bufSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rhsDevice, bufSize, rhsHost, bufSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, bufSize, outHost, bufSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchMixed_external_vadd_kernel(lhsDevice, rhsDevice, outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outHost, bufSize, outDevice, bufSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v3.bin", outHost, bufSize), "./v3.bin");

cleanup:
  aclrtFree(lhsDevice);
  aclrtFree(rhsDevice);
  aclrtFree(outDevice);
  aclrtFreeHost(lhsHost);
  aclrtFreeHost(rhsHost);
  aclrtFreeHost(outHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
