// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/compare-select/vsel-f32-plds-us-pintlv-pbitcast
// -----------------------------------------------------------------------------
#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      const char *_recent = aclGetRecentErrMsg();                                \
      if (_recent != nullptr && _recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchVsel_f32_plds_us_pintlv_pbitcast_kernel_2d(float *v1, float *v2,
                                                      unsigned char *v3,
                                                      float *v4,
                                                      void *stream);

int main() {
  size_t elemCount_v1 = 128;
  size_t fileSize_v1 = elemCount_v1 * sizeof(float);
  size_t elemCount_v2 = 128;
  size_t fileSize_v2 = elemCount_v2 * sizeof(float);
  size_t elemCount_v3 = 32;
  size_t fileSize_v3 = elemCount_v3 * sizeof(unsigned char);
  size_t elemCount_v4 = 128;
  size_t fileSize_v4 = elemCount_v4 * sizeof(float);
  float *v1Host = nullptr;
  float *v1Device = nullptr;
  float *v2Host = nullptr;
  float *v2Device = nullptr;
  unsigned char *v3Host = nullptr;
  unsigned char *v3Device = nullptr;
  float *v4Host = nullptr;
  float *v4Device = nullptr;

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

  ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSize_v1));
  ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSize_v2));
  ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSize_v3));
  ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSize_v4));
  ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize_v1, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize_v2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSize_v3, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSize_v4, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", fileSize_v1, v1Host, fileSize_v1);
  ReadFile("./v2.bin", fileSize_v2, v2Host, fileSize_v2);
  ReadFile("./v3.bin", fileSize_v3, v3Host, fileSize_v3);
  ReadFile("./v4.bin", fileSize_v4, v4Host, fileSize_v4);
  ACL_CHECK(aclrtMemcpy(v1Device, fileSize_v1, v1Host, fileSize_v1,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v2Device, fileSize_v2, v2Host, fileSize_v2,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v3Device, fileSize_v3, v3Host, fileSize_v3,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v4Device, fileSize_v4, v4Host, fileSize_v4,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchVsel_f32_plds_us_pintlv_pbitcast_kernel_2d(v1Device, v2Device, v3Device,
                                                   v4Device, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(v4Host, fileSize_v4, v4Device, fileSize_v4,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v4.bin", v4Host, fileSize_v4);

cleanup:
  aclrtFree(v1Device);
  aclrtFree(v2Device);
  aclrtFree(v3Device);
  aclrtFree(v4Device);
  aclrtFreeHost(v1Host);
  aclrtFreeHost(v2Host);
  aclrtFreeHost(v3Host);
  aclrtFreeHost(v4Host);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
