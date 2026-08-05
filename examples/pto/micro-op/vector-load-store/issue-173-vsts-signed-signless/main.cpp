// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/vector-load-store/issue-173-vsts-signed-signless
// family: micro-op/vector-load-store
// target_ops: pto.vlds, pto.vsts
// scenarios: signed-i16, signless-i16, same-module, issue-173-regression
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


void LaunchIssue173VstsSignedSignlessDeepMerged(int16_t * p0, int16_t * p1, int16_t * p2, int16_t * p3, void *stream);
int main() {
  constexpr size_t elemCount = 1024;
  constexpr size_t fileSize = elemCount * sizeof(int16_t);
  size_t inputFileSize = fileSize;

  int16_t *v1Host = nullptr;
  int16_t *v2Host = nullptr;
  int16_t *v3Host = nullptr;
  int16_t *v4Host = nullptr;
  int16_t *v1Device = nullptr;
  int16_t *v2Device = nullptr;
  int16_t *v3Device = nullptr;
  int16_t *v4Device = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSize));
  ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSize));

  ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));

  FILE_CHECK(ReadFile("./v1.bin", inputFileSize, v1Host, fileSize) &&
                 inputFileSize == fileSize,
             "./v1.bin");
  inputFileSize = fileSize;
  FILE_CHECK(ReadFile("./v2.bin", inputFileSize, v2Host, fileSize) &&
                 inputFileSize == fileSize,
             "./v2.bin");
  inputFileSize = fileSize;
  FILE_CHECK(ReadFile("./v3.bin", inputFileSize, v3Host, fileSize) &&
                 inputFileSize == fileSize,
             "./v3.bin");
  inputFileSize = fileSize;
  FILE_CHECK(ReadFile("./v4.bin", inputFileSize, v4Host, fileSize) &&
                 inputFileSize == fileSize,
             "./v4.bin");

  ACL_CHECK(aclrtMemcpy(v1Device, fileSize, v1Host, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v2Device, fileSize, v2Host, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v3Device, fileSize, v3Host, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v4Device, fileSize, v4Host, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchIssue173VstsSignedSignlessDeepMerged(
      v1Device,
      v2Device,
      v3Device,
      v4Device,
      stream
  );
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(v2Host, fileSize, v2Device, fileSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v4Host, fileSize, v4Device, fileSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  FILE_CHECK(WriteFile("./v2.bin", v2Host, fileSize), "./v2.bin");
  FILE_CHECK(WriteFile("./v4.bin", v4Host, fileSize), "./v4.bin");

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
