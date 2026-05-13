// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/vector-load-store/dma-copy-rearrange
// family: micro-op/vector-load-store
// target_ops: pto.copy_gm_to_ubuf, pto.mte_ub_ub, pto.copy_ubuf_to_gm
// scenarios: i16, ub-rearrange, permute-4x16-rows
// -----------------------------------------------------------------------------

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

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

#define FILE_CHECK(expr, path)                                                   \
  do {                                                                           \
    if (!(expr)) {                                                               \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",       \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchDma_copy_rearrange_kernel(int16_t *v1, int16_t *v2,
                                     int64_t n_burst, int64_t len_burst,
                                     int64_t src_gap, int64_t dst_gap,
                                     void *stream);

int main() {
  constexpr size_t elemCount = 64;
  constexpr size_t fileSize = elemCount * sizeof(int16_t);
  size_t inputFileSize = fileSize;

  int16_t *v1Host = nullptr;
  int16_t *v2Host = nullptr;
  int16_t *v1Device = nullptr;
  int16_t *v2Device = nullptr;
  const int64_t nBurst = 1;
  const int64_t lenBurst = 1;
  const int64_t srcGap = 0;
  const int64_t dstGap = 0;
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

  ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSize));
  ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSize));
  ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));

  FILE_CHECK(ReadFile("./v1.bin", inputFileSize, v1Host, fileSize) &&
                 inputFileSize == fileSize,
             "./v1.bin");
  inputFileSize = fileSize;
  FILE_CHECK(ReadFile("./v2.bin", inputFileSize, v2Host, fileSize) &&
                 inputFileSize == fileSize,
             "./v2.bin");

  ACL_CHECK(aclrtMemcpy(v1Device, fileSize, v1Host, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v2Device, fileSize, v2Host, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchDma_copy_rearrange_kernel(v1Device, v2Device, nBurst, lenBurst,
                                  srcGap, dstGap, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(v2Host, fileSize, v2Device, fileSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  FILE_CHECK(WriteFile("./v2.bin", v2Host, fileSize), "./v2.bin");

cleanup:
  aclrtFree(v1Device);
  aclrtFree(v2Device);
  aclrtFreeHost(v1Host);
  aclrtFreeHost(v2Host);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
