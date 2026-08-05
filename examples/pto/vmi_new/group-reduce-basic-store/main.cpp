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

void LaunchVmi_group_reduce_basic_store_kernel(float *src8, float *src16,
                                               float *src32, float *dst8,
                                               float *dst16, float *dst32,
                                               void *stream);

int main() {
  constexpr size_t kRows = 8;
  constexpr size_t kSrc8Elems = kRows * 8;
  constexpr size_t kSrc16Elems = kRows * 16;
  constexpr size_t kSrc32Elems = kRows * 32;
  constexpr size_t kOutputElems = kRows;
  size_t src8Bytes = kSrc8Elems * sizeof(float);
  size_t src16Bytes = kSrc16Elems * sizeof(float);
  size_t src32Bytes = kSrc32Elems * sizeof(float);
  size_t dstBytes = kOutputElems * sizeof(float);
  float *src8Host = nullptr;
  float *src16Host = nullptr;
  float *src32Host = nullptr;
  float *dst8Host = nullptr;
  float *dst16Host = nullptr;
  float *dst32Host = nullptr;
  float *src8Device = nullptr;
  float *src16Device = nullptr;
  float *src32Device = nullptr;
  float *dst8Device = nullptr;
  float *dst16Device = nullptr;
  float *dst32Device = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&src8Host), src8Bytes));
  ACL_CHECK(aclrtMallocHost((void **)(&src16Host), src16Bytes));
  ACL_CHECK(aclrtMallocHost((void **)(&src32Host), src32Bytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dst8Host), dstBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dst16Host), dstBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dst32Host), dstBytes));
  ACL_CHECK(aclrtMalloc((void **)&src8Device, src8Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&src16Device, src16Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&src32Device, src32Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dst8Device, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dst16Device, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dst32Device, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", src8Bytes, src8Host, src8Bytes);
  ReadFile("./v2.bin", src16Bytes, src16Host, src16Bytes);
  ReadFile("./v3.bin", src32Bytes, src32Host, src32Bytes);
  ReadFile("./v4.bin", dstBytes, dst8Host, dstBytes);
  ReadFile("./v5.bin", dstBytes, dst16Host, dstBytes);
  ReadFile("./v6.bin", dstBytes, dst32Host, dstBytes);
  ACL_CHECK(aclrtMemcpy(src8Device, src8Bytes, src8Host, src8Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(src16Device, src16Bytes, src16Host, src16Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(src32Device, src32Bytes, src32Host, src32Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dst8Device, dstBytes, dst8Host, dstBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dst16Device, dstBytes, dst16Host, dstBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dst32Device, dstBytes, dst32Host, dstBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_group_reduce_basic_store_kernel(
      src8Device, src16Device, src32Device, dst8Device, dst16Device,
      dst32Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dst8Host, dstBytes, dst8Device, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(dst16Host, dstBytes, dst16Device, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(dst32Host, dstBytes, dst32Device, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v4.bin", dst8Host, dstBytes);
  WriteFile("./v5.bin", dst16Host, dstBytes);
  WriteFile("./v6.bin", dst32Host, dstBytes);

cleanup:
  aclrtFree(src8Device);
  aclrtFree(src16Device);
  aclrtFree(src32Device);
  aclrtFree(dst8Device);
  aclrtFree(dst16Device);
  aclrtFree(dst32Device);
  aclrtFreeHost(src8Host);
  aclrtFreeHost(src16Host);
  aclrtFreeHost(src32Host);
  aclrtFreeHost(dst8Host);
  aclrtFreeHost(dst16Host);
  aclrtFreeHost(dst32Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
