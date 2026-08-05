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
#include <stdint.h>

using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { const aclError _ret = (expr); if (_ret != ACL_SUCCESS) { std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); rc = 1; goto cleanup; } } while (0)

void LaunchSimt_atomic_packed_core_kernel(uint32_t *h2, uint32_t *b2,
                                          void *stream);

int main() {
  size_t elemCount = 16;
  size_t fileSize = elemCount * sizeof(uint32_t);
  uint32_t *h2Host = nullptr;
  uint32_t *b2Host = nullptr;
  uint32_t *h2Device = nullptr;
  uint32_t *b2Device = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&h2Host), fileSize));
  ACL_CHECK(aclrtMallocHost((void **)(&b2Host), fileSize));
  ACL_CHECK(aclrtMalloc((void **)&h2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&b2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin", fileSize, h2Host, fileSize);
  ReadFile("./v2.bin", fileSize, b2Host, fileSize);
  ACL_CHECK(aclrtMemcpy(h2Device, fileSize, h2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(b2Device, fileSize, b2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchSimt_atomic_packed_core_kernel(h2Device, b2Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(h2Host, fileSize, h2Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(b2Host, fileSize, b2Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v1.bin", h2Host, fileSize);
  WriteFile("./v2.bin", b2Host, fileSize);

cleanup:
  aclrtFree(b2Device);
  aclrtFree(h2Device);
  aclrtFreeHost(b2Host);
  aclrtFreeHost(h2Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
