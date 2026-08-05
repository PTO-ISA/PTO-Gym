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

void LaunchVmi_dense_group_reduce_multi_consumer_kernel(float *src, float *sum,
                                                        float *copy,
                                                        void *stream);

int main() {
  constexpr size_t kRows = 8;
  constexpr size_t kGroupSize = 32;
  constexpr size_t kInputElems = kRows * kGroupSize;
  constexpr size_t kSumElems = kRows;
  constexpr size_t kCopyElems = kInputElems;
  size_t srcBytes = kInputElems * sizeof(float);
  size_t sumBytes = kSumElems * sizeof(float);
  size_t copyBytes = kCopyElems * sizeof(float);
  float *srcHost = nullptr;
  float *srcDevice = nullptr;
  float *sumHost = nullptr;
  float *sumDevice = nullptr;
  float *copyHost = nullptr;
  float *copyDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&sumHost), sumBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&copyHost), copyBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&sumDevice, sumBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&copyDevice, copyBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", sumBytes, sumHost, sumBytes);
  ReadFile("./v3.bin", copyBytes, copyHost, copyBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(sumDevice, sumBytes, sumHost, sumBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(copyDevice, copyBytes, copyHost, copyBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_dense_group_reduce_multi_consumer_kernel(srcDevice, sumDevice,
                                                     copyDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(sumHost, sumBytes, sumDevice, sumBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(copyHost, copyBytes, copyDevice, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", sumHost, sumBytes);
  WriteFile("./v3.bin", copyHost, copyBytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(sumDevice);
  aclrtFree(copyDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(sumHost);
  aclrtFreeHost(copyHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
