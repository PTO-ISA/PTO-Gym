// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError _ret = (expr);                                              \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   (int)_ret, __FILE__, __LINE__);                             \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

void LaunchVcgGroup(float *src, float *dst_add, float *dst_max,
                    float *dst_min, void *stream);

int main() {
  constexpr size_t kElems = 64;
  constexpr size_t kBytes = kElems * sizeof(float);

  float *srcHost = nullptr;
  float *addHost = nullptr;
  float *maxHost = nullptr;
  float *minHost = nullptr;
  float *srcDevice = nullptr;
  float *addDevice = nullptr;
  float *maxDevice = nullptr;
  float *minDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t fileSize = kBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)&srcHost, kBytes));
  ACL_CHECK(aclrtMallocHost((void **)&addHost, kBytes));
  ACL_CHECK(aclrtMallocHost((void **)&maxHost, kBytes));
  ACL_CHECK(aclrtMallocHost((void **)&minHost, kBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&addDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&maxDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&minDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  if (!ReadFile("./src.bin", fileSize, srcHost, kBytes) || fileSize != kBytes) {
    std::fprintf(stderr, "[ERROR] failed to read src.bin\n");
    rc = 1;
    goto cleanup;
  }
  std::fill_n(addHost, kElems, 0.0f);
  std::fill_n(maxHost, kElems, 0.0f);
  std::fill_n(minHost, kElems, 0.0f);

  ACL_CHECK(aclrtMemcpy(srcDevice, kBytes, srcHost, kBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(addDevice, kBytes, addHost, kBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(maxDevice, kBytes, maxHost, kBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(minDevice, kBytes, minHost, kBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchVcgGroup(srcDevice, addDevice, maxDevice, minDevice, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(addHost, kBytes, addDevice, kBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(maxHost, kBytes, maxDevice, kBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(minHost, kBytes, minDevice, kBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./out_add.bin", addHost, kBytes);
  WriteFile("./out_max.bin", maxHost, kBytes);
  WriteFile("./out_min.bin", minHost, kBytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(addDevice);
  aclrtFree(maxDevice);
  aclrtFree(minDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(addHost);
  aclrtFreeHost(maxHost);
  aclrtFreeHost(minHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
