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

void LaunchFixpipe_relu_ub_cv_kernel(__fp16 *src, __fp16 *id, __fp16 *outUbRelu,
                                     __fp16 *outUbClip, __fp16 *outGmRelu,
                                     __fp16 *outGmClip, __fp16 *outL1Relu,
                                     __fp16 *outL1Clip, void *stream);

int main() {
  constexpr size_t kSrcElems = 50 * 64;
  constexpr size_t kIdElems = 40 * 50;
  constexpr size_t kOutElems = 40 * 64;
  constexpr size_t kSrcSize = kSrcElems * sizeof(__fp16);
  constexpr size_t kIdSize = kIdElems * sizeof(__fp16);
  constexpr size_t kOutSize = kOutElems * sizeof(__fp16);

  __fp16 *srcHost = nullptr;
  __fp16 *idHost = nullptr;
  __fp16 *outUbReluHost = nullptr;
  __fp16 *outUbClipHost = nullptr;
  __fp16 *outGmReluHost = nullptr;
  __fp16 *outGmClipHost = nullptr;
  __fp16 *outL1ReluHost = nullptr;
  __fp16 *outL1ClipHost = nullptr;
  __fp16 *srcDevice = nullptr;
  __fp16 *idDevice = nullptr;
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

  ACL_CHECK(aclrtMallocHost((void **)&srcHost, kSrcSize));
  ACL_CHECK(aclrtMallocHost((void **)&idHost, kIdSize));
  ACL_CHECK(aclrtMallocHost((void **)&outUbReluHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outUbClipHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outGmReluHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outGmClipHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outL1ReluHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outL1ClipHost, kOutSize));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kSrcSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&idDevice, kIdSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbReluDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbClipDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmReluDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmClipDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1ReluDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1ClipDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kIdSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, idHost, kIdSize) && inputSize == kIdSize,
             "./v1.bin");
  inputSize = kSrcSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, srcHost, kSrcSize) && inputSize == kSrcSize,
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
    inputSize = kOutSize;
    char path[16];
    std::snprintf(path, sizeof(path), "./v%d.bin", index);
    FILE_CHECK(ReadFile(path, inputSize, hostBuf, kOutSize) && inputSize == kOutSize,
               path);
  }

  ACL_CHECK(aclrtMemcpy(srcDevice, kSrcSize, srcHost, kSrcSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(idDevice, kIdSize, idHost, kIdSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbReluDevice, kOutSize, outUbReluHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbClipDevice, kOutSize, outUbClipHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmReluDevice, kOutSize, outGmReluHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmClipDevice, kOutSize, outGmClipHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1ReluDevice, kOutSize, outL1ReluHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1ClipDevice, kOutSize, outL1ClipHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_relu_ub_cv_kernel(srcDevice, idDevice, outUbReluDevice,
                                  outUbClipDevice, outGmReluDevice,
                                  outGmClipDevice, outL1ReluDevice,
                                  outL1ClipDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outUbReluHost, kOutSize, outUbReluDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outUbClipHost, kOutSize, outUbClipDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmReluHost, kOutSize, outGmReluDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmClipHost, kOutSize, outGmClipDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1ReluHost, kOutSize, outL1ReluDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1ClipHost, kOutSize, outL1ClipDevice, kOutSize,
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
    FILE_CHECK(WriteFile(path, hostBuf, kOutSize), path);
  }

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(idDevice);
  aclrtFree(outUbReluDevice);
  aclrtFree(outUbClipDevice);
  aclrtFree(outGmReluDevice);
  aclrtFree(outGmClipDevice);
  aclrtFree(outL1ReluDevice);
  aclrtFree(outL1ClipDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(idHost);
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
