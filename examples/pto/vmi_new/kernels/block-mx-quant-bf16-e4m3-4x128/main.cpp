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

void LaunchVmi_block_mx_quant_bf16_e4m3_4x128_kernel(
    uint16_t *src, uint8_t *out, uint8_t *scale1, uint8_t *scale2,
    void *stream);

int main() {
  constexpr size_t kRows = 4;
  constexpr size_t kCols = 128;
  constexpr size_t kElems = kRows * kCols;
  constexpr size_t kScale1Bytes = 16;
  constexpr size_t kScale2Bytes = 256;
  size_t srcBytes = kElems * sizeof(uint16_t);
  size_t outBytes = kElems * sizeof(uint8_t);
  size_t scale1Bytes = kScale1Bytes;
  size_t scale2Bytes = kScale2Bytes;
  uint16_t *srcHost = nullptr;
  uint8_t *outHost = nullptr;
  uint8_t *scale1Host = nullptr;
  uint8_t *scale2Host = nullptr;
  uint16_t *srcDevice = nullptr;
  uint8_t *outDevice = nullptr;
  uint8_t *scale1Device = nullptr;
  uint8_t *scale2Device = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&outHost), outBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&scale1Host), scale1Bytes));
  ACL_CHECK(aclrtMallocHost((void **)(&scale2Host), scale2Bytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&scale1Device, scale1Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&scale2Device, scale2Bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", outBytes, outHost, outBytes);
  ReadFile("./v3.bin", scale1Bytes, scale1Host, scale1Bytes);
  ReadFile("./v4.bin", scale2Bytes, scale2Host, scale2Bytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, outBytes, outHost, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(scale1Device, scale1Bytes, scale1Host, scale1Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(scale2Device, scale2Bytes, scale2Host, scale2Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_block_mx_quant_bf16_e4m3_4x128_kernel(
      srcDevice, outDevice, scale1Device, scale2Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(scale1Host, scale1Bytes, scale1Device, scale1Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(scale2Host, scale2Bytes, scale2Device, scale2Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", outHost, outBytes);
  WriteFile("./v3.bin", scale1Host, scale1Bytes);
  WriteFile("./v4.bin", scale2Host, scale2Bytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(outDevice);
  aclrtFree(scale1Device);
  aclrtFree(scale2Device);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(outHost);
  aclrtFreeHost(scale1Host);
  aclrtFreeHost(scale2Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
