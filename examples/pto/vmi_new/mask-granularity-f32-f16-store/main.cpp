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

void LaunchVmi_mask_granularity_f32_f16_store_kernel(float *src, float *out32,
                                                     uint16_t *out16,
                                                     void *stream);

int main() {
  constexpr size_t kElems = 128;
  size_t srcBytes = kElems * sizeof(float);
  size_t out32Bytes = kElems * sizeof(float);
  size_t out16Bytes = kElems * sizeof(uint16_t);
  float *srcHost = nullptr;
  float *out32Host = nullptr;
  uint16_t *out16Host = nullptr;
  float *srcDevice = nullptr;
  float *out32Device = nullptr;
  uint16_t *out16Device = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&out32Host), out32Bytes));
  ACL_CHECK(aclrtMallocHost((void **)(&out16Host), out16Bytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&out32Device, out32Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&out16Device, out16Bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", out32Bytes, out32Host, out32Bytes);
  ReadFile("./v3.bin", out16Bytes, out16Host, out16Bytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(out32Device, out32Bytes, out32Host, out32Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(out16Device, out16Bytes, out16Host, out16Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVmi_mask_granularity_f32_f16_store_kernel(srcDevice, out32Device,
                                                  out16Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(out32Host, out32Bytes, out32Device, out32Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(out16Host, out16Bytes, out16Device, out16Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", out32Host, out32Bytes);
  WriteFile("./v3.bin", out16Host, out16Bytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(out32Device);
  aclrtFree(out16Device);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(out32Host);
  aclrtFreeHost(out16Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
