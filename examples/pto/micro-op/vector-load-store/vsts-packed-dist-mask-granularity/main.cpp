// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/vector-load-store/vsts-packed-dist-mask-granularity
// family: micro-op/vector-load-store
// target_ops: pto.vlds, pto.vsts
// scenarios: packed-store, pk-b64, pk4-b32, b32-mask-granularity
// -----------------------------------------------------------------------------

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#ifndef TMRGSORT_HPP
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif

#define ACL_CHECK(expr)                                                         \
  do {                                                                          \
    const aclError _ret = (expr);                                               \
    if (_ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   (int)_ret, __FILE__, __LINE__);                              \
      const char *_recent = aclGetRecentErrMsg();                               \
      if (_recent != nullptr && _recent[0] != '\0')                             \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);           \
      rc = 1;                                                                   \
      goto cleanup;                                                             \
    }                                                                           \
  } while (0)

void LaunchVstsPackedDistMaskGranularityKernel(int64_t *v1, float *v2,
                                               uint8_t *v3, uint8_t *v4,
                                               void *stream);

int main() {
  size_t fileSizeV1 = 1024 * sizeof(int64_t);
  size_t fileSizeV2 = 512 * sizeof(float);
  size_t fileSizeV3 = 256 * sizeof(uint8_t);
  size_t fileSizeV4 = 256 * sizeof(uint8_t);

  int64_t *v1Host = nullptr;
  float *v2Host = nullptr;
  uint8_t *v3Host = nullptr;
  uint8_t *v4Host = nullptr;
  int64_t *v1Device = nullptr;
  float *v2Device = nullptr;
  uint8_t *v3Device = nullptr;
  uint8_t *v4Device = nullptr;

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

  ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSizeV1));
  ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSizeV2));
  ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSizeV3));
  ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSizeV4));
  ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSizeV1, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSizeV2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSizeV3, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSizeV4, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", fileSizeV1, v1Host, fileSizeV1);
  ReadFile("./v2.bin", fileSizeV2, v2Host, fileSizeV2);
  ReadFile("./v3.bin", fileSizeV3, v3Host, fileSizeV3);
  ReadFile("./v4.bin", fileSizeV4, v4Host, fileSizeV4);
  ACL_CHECK(aclrtMemcpy(v1Device, fileSizeV1, v1Host, fileSizeV1, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v2Device, fileSizeV2, v2Host, fileSizeV2, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v3Device, fileSizeV3, v3Host, fileSizeV3, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v4Device, fileSizeV4, v4Host, fileSizeV4, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchVstsPackedDistMaskGranularityKernel(v1Device, v2Device, v3Device,
                                            v4Device, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(v2Host, fileSizeV2, v2Device, fileSizeV2, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v4Host, fileSizeV4, v4Device, fileSizeV4, ACL_MEMCPY_DEVICE_TO_HOST));

  WriteFile("./v2.bin", v2Host, fileSizeV2);
  WriteFile("./v4.bin", v4Host, fileSizeV4);

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
