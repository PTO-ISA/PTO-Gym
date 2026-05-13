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

void LaunchFixpipe_quant_clip_f16_ub_cv_kernel(
    __fp16 *src, __fp16 *id, uint32_t *fp, __fp16 *outUbQuant,
    __fp16 *outUbClip, __fp16 *outGmQuant, __fp16 *outGmClip,
    __fp16 *outL1Quant, __fp16 *outL1Clip, void *stream);

int main() {
  constexpr size_t kSizeSrcElems = 50 * 64;
  constexpr size_t kSizeIdElems = 40 * 50;
  constexpr size_t kFpElems = 128;
  constexpr size_t kOutElems = 40 * 64;
  constexpr size_t kSizeSrc = kSizeSrcElems * sizeof(__fp16);
  constexpr size_t kSizeId = kSizeIdElems * sizeof(__fp16);
  constexpr size_t kSizeFp = kFpElems * sizeof(uint32_t);
  constexpr size_t kSizeOut = kOutElems * sizeof(__fp16);

  __fp16 *srcHost = nullptr;
  __fp16 *idHost = nullptr;
  uint32_t *fpHost = nullptr;
  __fp16 *outUbQuantHost = nullptr;
  __fp16 *outUbClipHost = nullptr;
  __fp16 *outGmQuantHost = nullptr;
  __fp16 *outGmClipHost = nullptr;
  __fp16 *outL1QuantHost = nullptr;
  __fp16 *outL1ClipHost = nullptr;
  __fp16 *srcDevice = nullptr;
  __fp16 *idDevice = nullptr;
  uint32_t *fpDevice = nullptr;
  __fp16 *outUbQuantDevice = nullptr;
  __fp16 *outUbClipDevice = nullptr;
  __fp16 *outGmQuantDevice = nullptr;
  __fp16 *outGmClipDevice = nullptr;
  __fp16 *outL1QuantDevice = nullptr;
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

  ACL_CHECK(aclrtMallocHost((void **)&srcHost, kSizeSrc));
  ACL_CHECK(aclrtMallocHost((void **)&idHost, kSizeId));
  ACL_CHECK(aclrtMallocHost((void **)&fpHost, kSizeFp));
  ACL_CHECK(aclrtMallocHost((void **)&outUbQuantHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outUbClipHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outGmQuantHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outGmClipHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outL1QuantHost, kSizeOut));
  ACL_CHECK(aclrtMallocHost((void **)&outL1ClipHost, kSizeOut));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kSizeSrc, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&idDevice, kSizeId, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&fpDevice, kSizeFp, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbQuantDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbClipDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmQuantDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmClipDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1QuantDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1ClipDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kSizeId;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, idHost, kSizeId) && inputSize == kSizeId,
             "./v1.bin");
  inputSize = kSizeSrc;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, srcHost, kSizeSrc) && inputSize == kSizeSrc,
             "./v2.bin");
  inputSize = kSizeFp;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, fpHost, kSizeFp) && inputSize == kSizeFp,
             "./v3.bin");
  for (int index = 4; index <= 9; ++index) {
    __fp16 *hostBuf = nullptr;
    switch (index) {
    case 4: hostBuf = outUbQuantHost; break;
    case 5: hostBuf = outUbClipHost; break;
    case 6: hostBuf = outGmQuantHost; break;
    case 7: hostBuf = outGmClipHost; break;
    case 8: hostBuf = outL1QuantHost; break;
    case 9: hostBuf = outL1ClipHost; break;
    }
    inputSize = kSizeOut;
    char path[16];
    std::snprintf(path, sizeof(path), "./v%d.bin", index);
    FILE_CHECK(ReadFile(path, inputSize, hostBuf, kSizeOut) && inputSize == kSizeOut,
               path);
  }

  ACL_CHECK(aclrtMemcpy(srcDevice, kSizeSrc, srcHost, kSizeSrc, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(idDevice, kSizeId, idHost, kSizeId, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(fpDevice, kSizeFp, fpHost, kSizeFp, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbQuantDevice, kSizeOut, outUbQuantHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbClipDevice, kSizeOut, outUbClipHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmQuantDevice, kSizeOut, outGmQuantHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmClipDevice, kSizeOut, outGmClipHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1QuantDevice, kSizeOut, outL1QuantHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1ClipDevice, kSizeOut, outL1ClipHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_quant_clip_f16_ub_cv_kernel(
      srcDevice, idDevice, fpDevice, outUbQuantDevice, outUbClipDevice,
      outGmQuantDevice, outGmClipDevice, outL1QuantDevice, outL1ClipDevice,
      stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outUbQuantHost, kSizeOut, outUbQuantDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outUbClipHost, kSizeOut, outUbClipDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmQuantHost, kSizeOut, outGmQuantDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmClipHost, kSizeOut, outGmClipDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1QuantHost, kSizeOut, outL1QuantDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1ClipHost, kSizeOut, outL1ClipDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  for (int index = 4; index <= 9; ++index) {
    __fp16 *hostBuf = nullptr;
    switch (index) {
    case 4: hostBuf = outUbQuantHost; break;
    case 5: hostBuf = outUbClipHost; break;
    case 6: hostBuf = outGmQuantHost; break;
    case 7: hostBuf = outGmClipHost; break;
    case 8: hostBuf = outL1QuantHost; break;
    case 9: hostBuf = outL1ClipHost; break;
    }
    char path[16];
    std::snprintf(path, sizeof(path), "./v%d.bin", index);
    FILE_CHECK(WriteFile(path, hostBuf, kSizeOut), path);
  }

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(idDevice);
  aclrtFree(fpDevice);
  aclrtFree(outUbQuantDevice);
  aclrtFree(outUbClipDevice);
  aclrtFree(outGmQuantDevice);
  aclrtFree(outGmClipDevice);
  aclrtFree(outL1QuantDevice);
  aclrtFree(outL1ClipDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(idHost);
  aclrtFreeHost(fpHost);
  aclrtFreeHost(outUbQuantHost);
  aclrtFreeHost(outUbClipHost);
  aclrtFreeHost(outGmQuantHost);
  aclrtFreeHost(outGmClipHost);
  aclrtFreeHost(outL1QuantHost);
  aclrtFreeHost(outL1ClipHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
