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

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError _ret = (expr);                                              \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   (int)_ret, __FILE__, __LINE__);                             \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

void LaunchVmi_interleave_layout_roundtrip_kernel(
    uint16_t *lhs, uint16_t *rhs, float *intlvLow, float *intlvHigh,
    float *roundtripLow, float *roundtripHigh, float *denseLhs,
    float *denseRhs, float *denseRoundtripLow, float *denseRoundtripHigh,
    void *stream);

int main() {
  constexpr size_t kWideElems = 256;
  constexpr size_t kDenseElems = 128;
  constexpr size_t kBuffers = 10;
  const size_t sizes[kBuffers] = {
      kWideElems * sizeof(uint16_t), kWideElems * sizeof(uint16_t),
      kWideElems * sizeof(float),    kWideElems * sizeof(float),
      kWideElems * sizeof(float),    kWideElems * sizeof(float),
      kDenseElems * sizeof(float),   kDenseElems * sizeof(float),
      kDenseElems * sizeof(float),   kDenseElems * sizeof(float)};
  void *host[kBuffers] = {};
  void *device[kBuffers] = {};
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

  for (size_t i = 0; i < kBuffers; ++i) {
    ACL_CHECK(aclrtMallocHost(&host[i], sizes[i]));
    ACL_CHECK(aclrtMalloc(&device[i], sizes[i],
                          ACL_MEM_MALLOC_HUGE_FIRST));
    char path[32];
    std::snprintf(path, sizeof(path), "./v%zu.bin", i + 1);
    size_t fileSize = sizes[i];
    ReadFile(path, fileSize, host[i], sizes[i]);
    ACL_CHECK(aclrtMemcpy(device[i], sizes[i], host[i], sizes[i],
                          ACL_MEMCPY_HOST_TO_DEVICE));
  }

  LaunchVmi_interleave_layout_roundtrip_kernel(
      static_cast<uint16_t *>(device[0]), static_cast<uint16_t *>(device[1]),
      static_cast<float *>(device[2]), static_cast<float *>(device[3]),
      static_cast<float *>(device[4]), static_cast<float *>(device[5]),
      static_cast<float *>(device[6]), static_cast<float *>(device[7]),
      static_cast<float *>(device[8]), static_cast<float *>(device[9]), stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = 2; i < kBuffers; ++i) {
    ACL_CHECK(aclrtMemcpy(host[i], sizes[i], device[i], sizes[i],
                          ACL_MEMCPY_DEVICE_TO_HOST));
    char path[32];
    std::snprintf(path, sizeof(path), "./v%zu.bin", i + 1);
    WriteFile(path, host[i], sizes[i]);
  }

cleanup:
  for (void *ptr : device)
    aclrtFree(ptr);
  for (void *ptr : host)
    aclrtFreeHost(ptr);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
