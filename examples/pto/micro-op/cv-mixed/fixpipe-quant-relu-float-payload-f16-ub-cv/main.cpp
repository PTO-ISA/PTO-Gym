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
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",        \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchFixpipe_quant_relu_float_payload_f16_ub_cv_kernel(
    __fp16 *lhs, __fp16 *rhs, __fp16 *outUb, __fp16 *outGm, __fp16 *outL1,
    void *stream);

int main() {
  constexpr size_t kLhsElems = 40 * 50;
  constexpr size_t kRhsElems = 50 * 64;
  constexpr size_t kOutElems = 40 * 64;
  constexpr size_t kLhsSize = kLhsElems * sizeof(__fp16);
  constexpr size_t kRhsSize = kRhsElems * sizeof(__fp16);
  constexpr size_t kOutSize = kOutElems * sizeof(__fp16);

  __fp16 *lhsHost = nullptr;
  __fp16 *rhsHost = nullptr;
  __fp16 *outUbHost = nullptr;
  __fp16 *outGmHost = nullptr;
  __fp16 *outL1Host = nullptr;
  __fp16 *lhsDevice = nullptr;
  __fp16 *rhsDevice = nullptr;
  __fp16 *outUbDevice = nullptr;
  __fp16 *outGmDevice = nullptr;
  __fp16 *outL1Device = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t inputSize = 0;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)&lhsHost, kLhsSize));
  ACL_CHECK(aclrtMallocHost((void **)&rhsHost, kRhsSize));
  ACL_CHECK(aclrtMallocHost((void **)&outUbHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outGmHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outL1Host, kOutSize));
  ACL_CHECK(aclrtMalloc((void **)&lhsDevice, kLhsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&rhsDevice, kRhsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1Device, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kLhsSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, lhsHost, kLhsSize) && inputSize == kLhsSize,
             "./v1.bin");
  inputSize = kRhsSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, rhsHost, kRhsSize) && inputSize == kRhsSize,
             "./v2.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, outUbHost, kOutSize) && inputSize == kOutSize,
             "./v3.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v4.bin", inputSize, outGmHost, kOutSize) && inputSize == kOutSize,
             "./v4.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v5.bin", inputSize, outL1Host, kOutSize) && inputSize == kOutSize,
             "./v5.bin");

  ACL_CHECK(aclrtMemcpy(lhsDevice, kLhsSize, lhsHost, kLhsSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rhsDevice, kRhsSize, rhsHost, kRhsSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbDevice, kOutSize, outUbHost, kOutSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmDevice, kOutSize, outGmHost, kOutSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1Device, kOutSize, outL1Host, kOutSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_quant_relu_float_payload_f16_ub_cv_kernel(
      lhsDevice, rhsDevice, outUbDevice, outGmDevice, outL1Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outUbHost, kOutSize, outUbDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmHost, kOutSize, outGmDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1Host, kOutSize, outL1Device, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v3.bin", outUbHost, kOutSize), "./v3.bin");
  FILE_CHECK(WriteFile("./v4.bin", outGmHost, kOutSize), "./v4.bin");
  FILE_CHECK(WriteFile("./v5.bin", outL1Host, kOutSize), "./v5.bin");

cleanup:
  aclrtFree(lhsDevice);
  aclrtFree(rhsDevice);
  aclrtFree(outUbDevice);
  aclrtFree(outGmDevice);
  aclrtFree(outL1Device);
  aclrtFreeHost(lhsHost);
  aclrtFreeHost(rhsHost);
  aclrtFreeHost(outUbHost);
  aclrtFreeHost(outGmHost);
  aclrtFreeHost(outL1Host);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
