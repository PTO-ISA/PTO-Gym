// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "test_common.h"
#include "acl/acl.h"
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

void LaunchVcvt_low_precision_roundtrip_kernel(
    uint8_t *f8e4_in, uint8_t *f8e5_in, uint8_t *hif8_in,
    uint8_t *f4e1_in, uint8_t *f4e2_in, uint8_t *f8e4_out,
    uint8_t *f8e5_out, uint8_t *hif8_out, uint8_t *f4e1_out,
    uint8_t *f4e2_out, void *stream);

int main() {
  constexpr size_t kBufferCount = 10;
  constexpr size_t kInputCount = 5;
  constexpr size_t kBytes = 1024;
  const char *files[kBufferCount] = {
      "v1_f8e4_in.bin",  "v2_f8e5_in.bin",  "v3_hif8_in.bin",
      "v4_f4e1_in.bin",  "v5_f4e2_in.bin",  "v6_f8e4_out.bin",
      "v7_f8e5_out.bin", "v8_hif8_out.bin", "v9_f4e1_out.bin",
      "v10_f4e2_out.bin"};

  uint8_t *host[kBufferCount] = {};
  uint8_t *device[kBufferCount] = {};

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

  for (size_t i = 0; i < kBufferCount; ++i) {
    ACL_CHECK(aclrtMallocHost((void **)(&host[i]), kBytes));
    ACL_CHECK(aclrtMalloc((void **)&device[i], kBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    size_t fileSize = kBytes;
    ReadFile(files[i], fileSize, host[i], kBytes);
    ACL_CHECK(aclrtMemcpy(device[i], kBytes, host[i], kBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
  }

  LaunchVcvt_low_precision_roundtrip_kernel(
      device[0], device[1], device[2], device[3], device[4], device[5],
      device[6], device[7], device[8], device[9], stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = kInputCount; i < kBufferCount; ++i) {
    ACL_CHECK(aclrtMemcpy(host[i], kBytes, device[i], kBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(files[i], host[i], kBytes);
  }

cleanup:
  for (size_t i = 0; i < kBufferCount; ++i) {
    aclrtFree(device[i]);
    aclrtFreeHost(host[i]);
  }
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
