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

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError _ret = (expr);                                              \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   (int)_ret, __FILE__, __LINE__);                             \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

void LaunchVmiVmullPairResult(int32_t *lhs, int32_t *rhs, uint32_t *low,
                              uint32_t *high, void *stream);

int main() {
  constexpr size_t kBufferCount = 4;
  constexpr size_t kBufferBytes = (64 + 128) * sizeof(uint32_t);
  const char *paths[kBufferCount] = {
      "./lhs.bin", "./rhs.bin", "./low.bin", "./high.bin",
  };
  const size_t sizes[kBufferCount] = {
      kBufferBytes, kBufferBytes, kBufferBytes, kBufferBytes,
  };
  void *host[kBufferCount] = {};
  void *device[kBufferCount] = {};
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

  for (size_t i = 0; i < kBufferCount; ++i) {
    ACL_CHECK(aclrtMallocHost(&host[i], sizes[i]));
    ACL_CHECK(aclrtMalloc(&device[i], sizes[i], ACL_MEM_MALLOC_HUGE_FIRST));
    size_t fileSize = sizes[i];
    if (!ReadFile(paths[i], fileSize, host[i], sizes[i]) ||
        fileSize != sizes[i]) {
      std::fprintf(stderr, "[ERROR] failed to read %s (%zu bytes)\n", paths[i],
                   sizes[i]);
      rc = 1;
      goto cleanup;
    }
    ACL_CHECK(aclrtMemcpy(device[i], sizes[i], host[i], sizes[i],
                         ACL_MEMCPY_HOST_TO_DEVICE));
  }

  LaunchVmiVmullPairResult(static_cast<int32_t *>(device[0]),
                           static_cast<int32_t *>(device[1]),
                           static_cast<uint32_t *>(device[2]),
                           static_cast<uint32_t *>(device[3]), stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = 2; i < kBufferCount; ++i) {
    ACL_CHECK(aclrtMemcpy(host[i], sizes[i], device[i], sizes[i],
                         ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile(paths[i], host[i], sizes[i])) {
      std::fprintf(stderr, "[ERROR] failed to write %s\n", paths[i]);
      rc = 1;
      goto cleanup;
    }
  }

cleanup:
  for (size_t i = 0; i < kBufferCount; ++i) {
    if (device[i])
      aclrtFree(device[i]);
    if (host[i])
      aclrtFreeHost(host[i]);
  }
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
