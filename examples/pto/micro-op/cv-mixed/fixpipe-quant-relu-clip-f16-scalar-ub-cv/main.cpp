// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>

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

void LaunchFixpipe_quant_relu_clip_f16_scalar_ub_cv_kernel(
    __fp16 *lhs, __fp16 *rhs, __fp16 *outUbRelu, __fp16 *outUbClip,
    __fp16 *outGmRelu, __fp16 *outGmClip, __fp16 *outL1Relu,
    __fp16 *outL1Clip, void *stream);

int main() {
  constexpr size_t kSizeLhsElems = 40 * 50;
  constexpr size_t kSizeRhsElems = 50 * 64;
  constexpr size_t kOutElems = 40 * 64;
  constexpr size_t kSizeLhs = kSizeLhsElems * sizeof(__fp16);
  constexpr size_t kSizeRhs = kSizeRhsElems * sizeof(__fp16);
  constexpr size_t kSizeOut = kOutElems * sizeof(__fp16);

  __fp16 *lhsHost = nullptr;
  __fp16 *rhsHost = nullptr;
  __fp16 *outUbReluHost = nullptr;
  __fp16 *outUbClipHost = nullptr;
  __fp16 *outGmReluHost = nullptr;
  __fp16 *outGmClipHost = nullptr;
  __fp16 *outL1ReluHost = nullptr;
  __fp16 *outL1ClipHost = nullptr;
  __fp16 *lhsDevice = nullptr;
  __fp16 *rhsDevice = nullptr;
  __fp16 *outUbReluDevice = nullptr;
  __fp16 *outUbClipDevice = nullptr;
  __fp16 *outGmReluDevice = nullptr;
  __fp16 *outGmClipDevice = nullptr;
  __fp16 *outL1ReluDevice = nullptr;
  __fp16 *outL1ClipDevice = nullptr;

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

  ACL_CHECK(aclrtMallocHost((void **)&lhsHost, kSizeLhs));
  ACL_CHECK(aclrtMallocHost((void **)&rhsHost, kSizeRhs));
  ACL_CHECK(aclrtMallocHost((void **)&outUbReluHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outUbClipHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outGmReluHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outGmClipHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outL1ReluHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outL1ClipHost, kSizeOut));
  ACL_CHECK(aclrtMalloc((void **)&lhsDevice, kSizeLhs, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&rhsDevice, kSizeRhs, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbReluDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbClipDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmReluDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmClipDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1ReluDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1ClipDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kSizeLhs;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, lhsHost, kSizeLhs) && inputSize == kSizeLhs,
             "./v1.bin");
  inputSize = kSizeRhs;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, rhsHost, kSizeRhs) && inputSize == kSizeRhs,
             "./v2.bin");
  for (int index = 3; index <= 8; ++index) {
    __fp16 *hostBuf = nullptr;
    switch (index) {
    case 3: hostBuf = outUbReluHost; break;
    case 4: hostBuf = outUbClipHost; break;
    case 5: hostBuf = outGmReluHost; break;
    case 6: hostBuf = outGmClipHost; break;
    case 7: hostBuf = outL1ReluHost; break;
    case 8: hostBuf = outL1ClipHost; break;
    }
    inputSize = kSizeOut;
    char path[16];
    std::snprintf(path, sizeof(path), "./v%d.bin", index);
    FILE_CHECK(ReadFile(path, inputSize, hostBuf, kSizeOut) && inputSize == kSizeOut,
               path);
  }

  ACL_CHECK(aclrtMemcpy(lhsDevice, kSizeLhs, lhsHost, kSizeLhs, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rhsDevice, kSizeRhs, rhsHost, kSizeRhs, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbReluDevice, kSizeOut, outUbReluHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbClipDevice, kSizeOut, outUbClipHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmReluDevice, kSizeOut, outGmReluHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmClipDevice, kSizeOut, outGmClipHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1ReluDevice, kSizeOut, outL1ReluHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1ClipDevice, kSizeOut, outL1ClipHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_quant_relu_clip_f16_scalar_ub_cv_kernel(
      lhsDevice, rhsDevice, outUbReluDevice, outUbClipDevice, outGmReluDevice,
      outGmClipDevice, outL1ReluDevice, outL1ClipDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outUbReluHost, kSizeOut, outUbReluDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outUbClipHost, kSizeOut, outUbClipDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmReluHost, kSizeOut, outGmReluDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmClipHost, kSizeOut, outGmClipDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1ReluHost, kSizeOut, outL1ReluDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1ClipHost, kSizeOut, outL1ClipDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  for (int index = 3; index <= 8; ++index) {
    __fp16 *hostBuf = nullptr;
    switch (index) {
    case 3: hostBuf = outUbReluHost; break;
    case 4: hostBuf = outUbClipHost; break;
    case 5: hostBuf = outGmReluHost; break;
    case 6: hostBuf = outGmClipHost; break;
    case 7: hostBuf = outL1ReluHost; break;
    case 8: hostBuf = outL1ClipHost; break;
    }
    char path[16];
    std::snprintf(path, sizeof(path), "./v%d.bin", index);
    FILE_CHECK(WriteFile(path, hostBuf, kSizeOut), path);
  }

cleanup:
  aclrtFree(lhsDevice);
  aclrtFree(rhsDevice);
  aclrtFree(outUbReluDevice);
  aclrtFree(outUbClipDevice);
  aclrtFree(outGmReluDevice);
  aclrtFree(outGmClipDevice);
  aclrtFree(outL1ReluDevice);
  aclrtFree(outL1ClipDevice);
  aclrtFreeHost(lhsHost);
  aclrtFreeHost(rhsHost);
  aclrtFreeHost(outUbReluHost);
  aclrtFreeHost(outUbClipHost);
  aclrtFreeHost(outGmReluHost);
  aclrtFreeHost(outGmClipHost);
  aclrtFreeHost(outL1ReluHost);
  aclrtFreeHost(outL1ClipHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
