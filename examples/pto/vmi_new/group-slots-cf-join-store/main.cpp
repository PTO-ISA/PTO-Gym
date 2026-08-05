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

void LaunchVmi_group_slots_cf_join_store_kernel(float *src, float *rhs,
                                                float *dstReduce,
                                                float *dstSlot, void *stream);

int main() {
  constexpr size_t kRows = 8;
  constexpr size_t kGroupSize = 16;
  constexpr size_t kInputElems = kRows * kGroupSize;
  constexpr size_t kOutputElems = kRows;
  size_t srcBytes = kInputElems * sizeof(float);
  size_t rhsBytes = kOutputElems * sizeof(float);
  size_t dstBytes = kOutputElems * sizeof(float);
  float *srcHost = nullptr;
  float *rhsHost = nullptr;
  float *dstReduceHost = nullptr;
  float *dstSlotHost = nullptr;
  float *srcDevice = nullptr;
  float *rhsDevice = nullptr;
  float *dstReduceDevice = nullptr;
  float *dstSlotDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&dstReduceHost), dstBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dstSlotHost), dstBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&rhsDevice, rhsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstReduceDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstSlotDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", rhsBytes, rhsHost, rhsBytes);
  ReadFile("./v3.bin", dstBytes, dstReduceHost, dstBytes);
  ReadFile("./v4.bin", dstBytes, dstSlotHost, dstBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rhsDevice, rhsBytes, rhsHost, rhsBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstReduceDevice, dstBytes, dstReduceHost, dstBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstSlotDevice, dstBytes, dstSlotHost, dstBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_group_slots_cf_join_store_kernel(srcDevice, rhsDevice,
                                             dstReduceDevice, dstSlotDevice,
                                             stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstReduceHost, dstBytes, dstReduceDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(dstSlotHost, dstBytes, dstSlotDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", dstReduceHost, dstBytes);
  WriteFile("./v4.bin", dstSlotHost, dstBytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(rhsDevice);
  aclrtFree(dstReduceDevice);
  aclrtFree(dstSlotDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(rhsHost);
  aclrtFreeHost(dstReduceHost);
  aclrtFreeHost(dstSlotHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
