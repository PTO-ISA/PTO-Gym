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

void LaunchFixpipe_relu_scalar_vector_f16_cv_kernel(
    __fp16 *lhs, __fp16 *rhs, uint32_t *reluFp, __fp16 *outUbScalar,
    __fp16 *outUbVector, __fp16 *outGmScalar, __fp16 *outGmVector,
    __fp16 *outL1Scalar, __fp16 *outL1Vector, void *stream);

int main() {
  constexpr size_t kLhsElems = 40 * 50;
  constexpr size_t kRhsElems = 50 * 64;
  constexpr size_t kReluFpElems = 64 * 2;
  constexpr size_t kOutElems = 40 * 64;
  constexpr size_t kLhsSize = kLhsElems * sizeof(__fp16);
  constexpr size_t kRhsSize = kRhsElems * sizeof(__fp16);
  constexpr size_t kReluFpSize = kReluFpElems * sizeof(uint32_t);
  constexpr size_t kOutSize = kOutElems * sizeof(__fp16);

  __fp16 *lhsHost = nullptr;
  __fp16 *rhsHost = nullptr;
  uint32_t *reluFpHost = nullptr;
  __fp16 *outUbScalarHost = nullptr;
  __fp16 *outUbVectorHost = nullptr;
  __fp16 *outGmScalarHost = nullptr;
  __fp16 *outGmVectorHost = nullptr;
  __fp16 *outL1ScalarHost = nullptr;
  __fp16 *outL1VectorHost = nullptr;
  __fp16 *lhsDevice = nullptr;
  __fp16 *rhsDevice = nullptr;
  uint32_t *reluFpDevice = nullptr;
  __fp16 *outUbScalarDevice = nullptr;
  __fp16 *outUbVectorDevice = nullptr;
  __fp16 *outGmScalarDevice = nullptr;
  __fp16 *outGmVectorDevice = nullptr;
  __fp16 *outL1ScalarDevice = nullptr;
  __fp16 *outL1VectorDevice = nullptr;

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
  ACL_CHECK(aclrtMallocHost((void **)&reluFpHost, kReluFpSize));
  ACL_CHECK(aclrtMallocHost((void **)&outUbScalarHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outUbVectorHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outGmScalarHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outGmVectorHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outL1ScalarHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outL1VectorHost, kOutSize));
  ACL_CHECK(aclrtMalloc((void **)&lhsDevice, kLhsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&rhsDevice, kRhsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&reluFpDevice, kReluFpSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbScalarDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbVectorDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmScalarDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmVectorDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1ScalarDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1VectorDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kLhsSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, lhsHost, kLhsSize) && inputSize == kLhsSize,
             "./v1.bin");
  inputSize = kRhsSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, rhsHost, kRhsSize) && inputSize == kRhsSize,
             "./v2.bin");
  inputSize = kReluFpSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, reluFpHost, kReluFpSize) &&
                 inputSize == kReluFpSize,
             "./v3.bin");

  for (int index = 4; index <= 9; ++index) {
    __fp16 *hostBuf = nullptr;
    switch (index) {
    case 4: hostBuf = outUbScalarHost; break;
    case 5: hostBuf = outUbVectorHost; break;
    case 6: hostBuf = outGmScalarHost; break;
    case 7: hostBuf = outGmVectorHost; break;
    case 8: hostBuf = outL1ScalarHost; break;
    case 9: hostBuf = outL1VectorHost; break;
    }
    inputSize = kOutSize;
    char path[16];
    std::snprintf(path, sizeof(path), "./v%d.bin", index);
    FILE_CHECK(ReadFile(path, inputSize, hostBuf, kOutSize) && inputSize == kOutSize,
               path);
  }

  ACL_CHECK(aclrtMemcpy(lhsDevice, kLhsSize, lhsHost, kLhsSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rhsDevice, kRhsSize, rhsHost, kRhsSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(reluFpDevice, kReluFpSize, reluFpHost, kReluFpSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbScalarDevice, kOutSize, outUbScalarHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbVectorDevice, kOutSize, outUbVectorHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmScalarDevice, kOutSize, outGmScalarHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmVectorDevice, kOutSize, outGmVectorHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1ScalarDevice, kOutSize, outL1ScalarHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1VectorDevice, kOutSize, outL1VectorHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_relu_scalar_vector_f16_cv_kernel(
      lhsDevice, rhsDevice, reluFpDevice, outUbScalarDevice, outUbVectorDevice,
      outGmScalarDevice, outGmVectorDevice, outL1ScalarDevice, outL1VectorDevice,
      stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outUbScalarHost, kOutSize, outUbScalarDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outUbVectorHost, kOutSize, outUbVectorDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmScalarHost, kOutSize, outGmScalarDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmVectorHost, kOutSize, outGmVectorDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1ScalarHost, kOutSize, outL1ScalarDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1VectorHost, kOutSize, outL1VectorDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  for (int index = 4; index <= 9; ++index) {
    __fp16 *hostBuf = nullptr;
    switch (index) {
    case 4: hostBuf = outUbScalarHost; break;
    case 5: hostBuf = outUbVectorHost; break;
    case 6: hostBuf = outGmScalarHost; break;
    case 7: hostBuf = outGmVectorHost; break;
    case 8: hostBuf = outL1ScalarHost; break;
    case 9: hostBuf = outL1VectorHost; break;
    }
    char path[16];
    std::snprintf(path, sizeof(path), "./v%d.bin", index);
    FILE_CHECK(WriteFile(path, hostBuf, kOutSize), path);
  }

cleanup:
  aclrtFree(lhsDevice);
  aclrtFree(rhsDevice);
  aclrtFree(reluFpDevice);
  aclrtFree(outUbScalarDevice);
  aclrtFree(outUbVectorDevice);
  aclrtFree(outGmScalarDevice);
  aclrtFree(outGmVectorDevice);
  aclrtFree(outL1ScalarDevice);
  aclrtFree(outL1VectorDevice);
  aclrtFreeHost(lhsHost);
  aclrtFreeHost(rhsHost);
  aclrtFreeHost(reluFpHost);
  aclrtFreeHost(outUbScalarHost);
  aclrtFreeHost(outUbVectorHost);
  aclrtFreeHost(outGmScalarHost);
  aclrtFreeHost(outGmVectorHost);
  aclrtFreeHost(outL1ScalarHost);
  aclrtFreeHost(outL1VectorHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
