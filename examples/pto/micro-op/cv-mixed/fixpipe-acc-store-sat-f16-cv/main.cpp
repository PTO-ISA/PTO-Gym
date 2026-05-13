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

void LaunchFixpipe_acc_store_sat_f16_cv_kernel(
    __fp16 *src, __fp16 *id, uint32_t *fp, __fp16 *outUbSat,
    __fp16 *outUbNosat, __fp16 *outGmSat, __fp16 *outGmNosat,
    __fp16 *outL1Sat, __fp16 *outL1Nosat, void *stream);

int main() {
  constexpr size_t kSrcElems = 50 * 64;
  constexpr size_t kIdElems = 40 * 50;
  constexpr size_t kFpElems = 128;
  constexpr size_t kOutElems = 40 * 64;
  constexpr size_t kSrcSize = kSrcElems * sizeof(__fp16);
  constexpr size_t kIdSize = kIdElems * sizeof(__fp16);
  constexpr size_t kFpSize = kFpElems * sizeof(uint32_t);
  constexpr size_t kOutSize = kOutElems * sizeof(__fp16);

  __fp16 *srcHost = nullptr;
  __fp16 *idHost = nullptr;
  uint32_t *fpHost = nullptr;
  __fp16 *outUbSatHost = nullptr;
  __fp16 *outUbNosatHost = nullptr;
  __fp16 *outGmSatHost = nullptr;
  __fp16 *outGmNosatHost = nullptr;
  __fp16 *outL1SatHost = nullptr;
  __fp16 *outL1NosatHost = nullptr;
  __fp16 *srcDevice = nullptr;
  __fp16 *idDevice = nullptr;
  uint32_t *fpDevice = nullptr;
  __fp16 *outUbSatDevice = nullptr;
  __fp16 *outUbNosatDevice = nullptr;
  __fp16 *outGmSatDevice = nullptr;
  __fp16 *outGmNosatDevice = nullptr;
  __fp16 *outL1SatDevice = nullptr;
  __fp16 *outL1NosatDevice = nullptr;

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
  ACL_CHECK(aclrtMallocHost((void **)&fpHost, kFpSize));
  ACL_CHECK(aclrtMallocHost((void **)&outUbSatHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outUbNosatHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outGmSatHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outGmNosatHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outL1SatHost, kOutSize));
  ACL_CHECK(aclrtMallocHost((void **)&outL1NosatHost, kOutSize));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kSrcSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&idDevice, kIdSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&fpDevice, kFpSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbSatDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outUbNosatDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmSatDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outGmNosatDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1SatDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outL1NosatDevice, kOutSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kIdSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, idHost, kIdSize) && inputSize == kIdSize,
             "./v1.bin");
  inputSize = kSrcSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, srcHost, kSrcSize) && inputSize == kSrcSize,
             "./v2.bin");
  inputSize = kFpSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, fpHost, kFpSize) && inputSize == kFpSize,
             "./v3.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v4.bin", inputSize, outUbSatHost, kOutSize) && inputSize == kOutSize,
             "./v4.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v5.bin", inputSize, outUbNosatHost, kOutSize) &&
                 inputSize == kOutSize,
             "./v5.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v6.bin", inputSize, outGmSatHost, kOutSize) && inputSize == kOutSize,
             "./v6.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v7.bin", inputSize, outGmNosatHost, kOutSize) &&
                 inputSize == kOutSize,
             "./v7.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v8.bin", inputSize, outL1SatHost, kOutSize) && inputSize == kOutSize,
             "./v8.bin");
  inputSize = kOutSize;
  FILE_CHECK(ReadFile("./v9.bin", inputSize, outL1NosatHost, kOutSize) &&
                 inputSize == kOutSize,
             "./v9.bin");

  ACL_CHECK(aclrtMemcpy(srcDevice, kSrcSize, srcHost, kSrcSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(idDevice, kIdSize, idHost, kIdSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(fpDevice, kFpSize, fpHost, kFpSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbSatDevice, kOutSize, outUbSatHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outUbNosatDevice, kOutSize, outUbNosatHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmSatDevice, kOutSize, outGmSatHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outGmNosatDevice, kOutSize, outGmNosatHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1SatDevice, kOutSize, outL1SatHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outL1NosatDevice, kOutSize, outL1NosatHost, kOutSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_acc_store_sat_f16_cv_kernel(
      srcDevice, idDevice, fpDevice, outUbSatDevice, outUbNosatDevice,
      outGmSatDevice, outGmNosatDevice, outL1SatDevice, outL1NosatDevice,
      stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outUbSatHost, kOutSize, outUbSatDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outUbNosatHost, kOutSize, outUbNosatDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmSatHost, kOutSize, outGmSatDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outGmNosatHost, kOutSize, outGmNosatDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1SatHost, kOutSize, outL1SatDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outL1NosatHost, kOutSize, outL1NosatDevice, kOutSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  FILE_CHECK(WriteFile("./v4.bin", outUbSatHost, kOutSize), "./v4.bin");
  FILE_CHECK(WriteFile("./v5.bin", outUbNosatHost, kOutSize), "./v5.bin");
  FILE_CHECK(WriteFile("./v6.bin", outGmSatHost, kOutSize), "./v6.bin");
  FILE_CHECK(WriteFile("./v7.bin", outGmNosatHost, kOutSize), "./v7.bin");
  FILE_CHECK(WriteFile("./v8.bin", outL1SatHost, kOutSize), "./v8.bin");
  FILE_CHECK(WriteFile("./v9.bin", outL1NosatHost, kOutSize), "./v9.bin");

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(idDevice);
  aclrtFree(fpDevice);
  aclrtFree(outUbSatDevice);
  aclrtFree(outUbNosatDevice);
  aclrtFree(outGmSatDevice);
  aclrtFree(outGmNosatDevice);
  aclrtFree(outL1SatDevice);
  aclrtFree(outL1NosatDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(idHost);
  aclrtFreeHost(fpHost);
  aclrtFreeHost(outUbSatHost);
  aclrtFreeHost(outUbNosatHost);
  aclrtFreeHost(outGmSatHost);
  aclrtFreeHost(outGmNosatHost);
  aclrtFreeHost(outL1SatHost);
  aclrtFreeHost(outL1NosatHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
