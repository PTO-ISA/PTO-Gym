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
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchVmi_group_slots_scf_for_store_kernel(float *init, float *src,
                                                float *dst, void *stream);

int main() {
  constexpr size_t kRows = 8;
  constexpr size_t kCols = 16;
  constexpr size_t kInitElems = kRows;
  constexpr size_t kSrcElems = kRows * kCols;
  constexpr size_t kDstElems = kRows;
  size_t initBytes = kInitElems * sizeof(float);
  size_t srcBytes = kSrcElems * sizeof(float);
  size_t dstBytes = kDstElems * sizeof(float);
  float *initHost = nullptr;
  float *srcHost = nullptr;
  float *dstHost = nullptr;
  float *initDevice = nullptr;
  float *srcDevice = nullptr;
  float *dstDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&initHost), initBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), srcBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dstHost), dstBytes));
  ACL_CHECK(aclrtMalloc((void **)&initDevice, initBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", initBytes, initHost, initBytes);
  ReadFile("./v2.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v3.bin", dstBytes, dstHost, dstBytes);
  ACL_CHECK(aclrtMemcpy(initDevice, initBytes, initHost, initBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstDevice, dstBytes, dstHost, dstBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_group_slots_scf_for_store_kernel(initDevice, srcDevice, dstDevice,
                                             stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstHost, dstBytes, dstDevice, dstBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", dstHost, dstBytes);

cleanup:
  aclrtFree(initDevice);
  aclrtFree(srcDevice);
  aclrtFree(dstDevice);
  aclrtFreeHost(initHost);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(dstHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
