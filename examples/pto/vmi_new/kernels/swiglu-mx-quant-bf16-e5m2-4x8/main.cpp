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

void LaunchVmi_swiglu_mx_quant_bf16_e5m2_4x8_kernel(
    uint16_t *src, uint8_t *out, uint8_t *scale, void *stream);

int main() {
  constexpr size_t kRows = 4;
  constexpr size_t kInputCols = 8;
  constexpr size_t kOutCols = 4;
  constexpr size_t kInputElems = kRows * kInputCols;
  constexpr size_t kOutElems = kRows * kOutCols;
  constexpr size_t kScaleBytes = kRows;
  size_t srcBytes = kInputElems * sizeof(uint16_t);
  size_t outBytes = kOutElems * sizeof(uint8_t);
  size_t scaleBytes = kScaleBytes;
  uint16_t *srcHost = nullptr;
  uint8_t *outHost = nullptr;
  uint8_t *scaleHost = nullptr;
  uint16_t *srcDevice = nullptr;
  uint8_t *outDevice = nullptr;
  uint8_t *scaleDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&scaleDevice, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", outBytes, outHost, outBytes);
  ReadFile("./v3.bin", scaleBytes, scaleHost, scaleBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, outBytes, outHost, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(scaleDevice, scaleBytes, scaleHost, scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_swiglu_mx_quant_bf16_e5m2_4x8_kernel(
      srcDevice, outDevice, scaleDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(scaleHost, scaleBytes, scaleDevice, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", outHost, outBytes);
  WriteFile("./v3.bin", scaleHost, scaleBytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(outDevice);
  aclrtFree(scaleDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(outHost);
  aclrtFreeHost(scaleHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
