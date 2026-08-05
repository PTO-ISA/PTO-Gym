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

void LaunchVmi_tquant_mxfp8_32x32_nd_kernel(float *src, uint8_t *out_fp8,
                                            uint8_t *out_e8m0, void *stream);

int main() {
  constexpr size_t kRows = 32;
  constexpr size_t kCols = 32;
  constexpr size_t kElems = kRows * kCols;
  constexpr size_t kE8m0Bytes = 256;
  size_t srcBytes = kElems * sizeof(float);
  size_t outFp8Bytes = kElems * sizeof(uint8_t);
  size_t outE8m0Bytes = kE8m0Bytes * sizeof(uint8_t);
  float *srcHost = nullptr;
  uint8_t *outFp8Host = nullptr;
  uint8_t *outE8m0Host = nullptr;
  float *srcDevice = nullptr;
  uint8_t *outFp8Device = nullptr;
  uint8_t *outE8m0Device = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&outFp8Host), outFp8Bytes));
  ACL_CHECK(aclrtMallocHost((void **)(&outE8m0Host), outE8m0Bytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outFp8Device, outFp8Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outE8m0Device, outE8m0Bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", outFp8Bytes, outFp8Host, outFp8Bytes);
  ReadFile("./v3.bin", outE8m0Bytes, outE8m0Host, outE8m0Bytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outFp8Device, outFp8Bytes, outFp8Host, outFp8Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outE8m0Device, outE8m0Bytes, outE8m0Host, outE8m0Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_tquant_mxfp8_32x32_nd_kernel(srcDevice, outFp8Device,
                                         outE8m0Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outFp8Host, outFp8Bytes, outFp8Device, outFp8Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outE8m0Host, outE8m0Bytes, outE8m0Device, outE8m0Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", outFp8Host, outFp8Bytes);
  WriteFile("./v3.bin", outE8m0Host, outE8m0Bytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(outFp8Device);
  aclrtFree(outE8m0Device);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(outFp8Host);
  aclrtFreeHost(outE8m0Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
