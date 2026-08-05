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

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchVmi_mask_select_store_kernel(float *src, float *rhs, float *dense,
                                        float *masked, void *stream);

int main() {
  constexpr size_t kElems = 64;
  size_t srcBytes = kElems * sizeof(float);
  size_t rhsBytes = kElems * sizeof(float);
  size_t denseBytes = kElems * sizeof(float);
  size_t maskedBytes = kElems * sizeof(float);
  float *srcHost = nullptr;
  float *rhsHost = nullptr;
  float *denseHost = nullptr;
  float *maskedHost = nullptr;
  float *srcDevice = nullptr;
  float *rhsDevice = nullptr;
  float *denseDevice = nullptr;
  float *maskedDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), srcBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&rhsHost), rhsBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&denseHost), denseBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&maskedHost), maskedBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&rhsDevice, rhsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&denseDevice, denseBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&maskedDevice, maskedBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", rhsBytes, rhsHost, rhsBytes);
  ReadFile("./v3.bin", denseBytes, denseHost, denseBytes);
  ReadFile("./v4.bin", maskedBytes, maskedHost, maskedBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rhsDevice, rhsBytes, rhsHost, rhsBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(denseDevice, denseBytes, denseHost, denseBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(maskedDevice, maskedBytes, maskedHost, maskedBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_mask_select_store_kernel(srcDevice, rhsDevice, denseDevice,
                                     maskedDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(denseHost, denseBytes, denseDevice, denseBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(maskedHost, maskedBytes, maskedDevice, maskedBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", denseHost, denseBytes);
  WriteFile("./v4.bin", maskedHost, maskedBytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(rhsDevice);
  aclrtFree(denseDevice);
  aclrtFree(maskedDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(rhsHost);
  aclrtFreeHost(denseHost);
  aclrtFreeHost(maskedHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
