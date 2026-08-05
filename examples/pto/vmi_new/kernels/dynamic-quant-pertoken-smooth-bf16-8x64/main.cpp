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

void LaunchVmi_dynamic_quant_pertoken_smooth_bf16_8x64_kernel(
    uint16_t *src, uint16_t *smooth, float *scale, uint8_t *out,
    void *stream);

int main() {
  constexpr size_t kRows = 8;
  constexpr size_t kCols = 64;
  constexpr size_t kElems = kRows * kCols;
  constexpr size_t kSmoothElems = kCols;
  constexpr size_t kScaleElems = 16;
  size_t srcBytes = kElems * sizeof(uint16_t);
  size_t smoothBytes = kSmoothElems * sizeof(uint16_t);
  size_t scaleBytes = kScaleElems * sizeof(float);
  size_t outBytes = kElems * sizeof(uint8_t);
  uint16_t *srcHost = nullptr;
  uint16_t *smoothHost = nullptr;
  float *scaleHost = nullptr;
  uint8_t *outHost = nullptr;
  uint16_t *srcDevice = nullptr;
  uint16_t *smoothDevice = nullptr;
  float *scaleDevice = nullptr;
  uint8_t *outDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&smoothHost), smoothBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&outHost), outBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&smoothDevice, smoothBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&scaleDevice, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", smoothBytes, smoothHost, smoothBytes);
  ReadFile("./v3.bin", scaleBytes, scaleHost, scaleBytes);
  ReadFile("./v4.bin", outBytes, outHost, outBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(smoothDevice, smoothBytes, smoothHost, smoothBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(scaleDevice, scaleBytes, scaleHost, scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, outBytes, outHost, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_dynamic_quant_pertoken_smooth_bf16_8x64_kernel(
      srcDevice, smoothDevice, scaleDevice, outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(scaleHost, scaleBytes, scaleDevice, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", scaleHost, scaleBytes);
  WriteFile("./v4.bin", outHost, outBytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(smoothDevice);
  aclrtFree(scaleDevice);
  aclrtFree(outDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(smoothHost);
  aclrtFreeHost(scaleHost);
  aclrtFreeHost(outHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
