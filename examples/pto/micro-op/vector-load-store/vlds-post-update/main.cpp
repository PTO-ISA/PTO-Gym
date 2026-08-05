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

void LaunchVldsPostUpdate(float *input, float *output, void *stream);

namespace {
constexpr size_t kElementCount = 1024;
constexpr size_t kBufferSize = kElementCount * sizeof(float);
}

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError ret = (expr);                                               \
    if (ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   static_cast<int>(ret), __FILE__, __LINE__);                \
      const char *recent = aclGetRecentErrMsg();                               \
      if (recent != nullptr && recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", recent);           \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

int main() {
  float *inputHost = nullptr;
  float *outputHost = nullptr;
  float *inputDevice = nullptr;
  float *outputDevice = nullptr;
  aclrtStream stream = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  size_t inputSize = kBufferSize;
  size_t outputSize = kBufferSize;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&inputHost), kBufferSize));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&outputHost), kBufferSize));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&outputDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input.bin", inputSize, inputHost, kBufferSize);
  ReadFile("./output.bin", outputSize, outputHost, kBufferSize);
  ACL_CHECK(aclrtMemcpy(inputDevice, kBufferSize, inputHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outputDevice, kBufferSize, outputHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchVldsPostUpdate(inputDevice, outputDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outputHost, kBufferSize, outputDevice, kBufferSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output.bin", outputHost, kBufferSize);

cleanup:
  aclrtFree(inputDevice);
  aclrtFree(outputDevice);
  aclrtFreeHost(inputHost);
  aclrtFreeHost(outputHost);
  if (stream != nullptr) {
    const aclError ret = aclrtDestroyStream(stream);
    if (ret != ACL_SUCCESS)
      std::fprintf(stderr, "[ERROR] aclrtDestroyStream failed: %d\n",
                   static_cast<int>(ret));
  }
  if (deviceSet) {
    const aclError ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS)
      std::fprintf(stderr, "[ERROR] aclrtResetDevice failed: %d\n",
                   static_cast<int>(ret));
  }
  if (aclInited) {
    const aclError ret = aclFinalize();
    if (ret != ACL_SUCCESS)
      std::fprintf(stderr, "[ERROR] aclFinalize failed: %d\n",
                   static_cast<int>(ret));
  }
  return rc;
}
