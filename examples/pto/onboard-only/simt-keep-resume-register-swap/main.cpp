// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "acl/acl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr size_t kThreads = 32;
constexpr size_t kOutputElements = 2 * kThreads;
constexpr int64_t kBase64A = 0x1122334400000000LL;
constexpr int64_t kBase64B = 0x5566778800000000LL;

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError error = (expr);                                             \
    if (error != ACL_SUCCESS) {                                                \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   static_cast<int>(error), __FILE__, __LINE__);               \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

} // namespace

void LaunchSimtKeepResumeRegisterSwapKernel(int32_t *output32,
                                            int64_t *output64, void *stream);

int main() {
  const size_t output32Bytes = kOutputElements * sizeof(int32_t);
  const size_t output64Bytes = kOutputElements * sizeof(int64_t);
  int32_t *hostOutput32 = nullptr;
  int64_t *hostOutput64 = nullptr;
  int32_t *deviceOutput32 = nullptr;
  int64_t *deviceOutput64 = nullptr;
  std::FILE *output32File = nullptr;
  std::FILE *output64File = nullptr;
  aclrtStream stream = nullptr;
  bool aclInitialized = false;
  bool deviceSet = false;
  int deviceId = 0;
  int rc = 0;

  ACL_CHECK(aclInit(nullptr));
  aclInitialized = true;
  if (const char *value = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(value);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&hostOutput32), output32Bytes));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&hostOutput64), output64Bytes));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceOutput32),
                        output32Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceOutput64),
                        output64Bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  for (size_t i = 0; i < kOutputElements; ++i) {
    hostOutput32[i] = -1;
    hostOutput64[i] = -1;
  }
  ACL_CHECK(aclrtMemcpy(deviceOutput32, output32Bytes, hostOutput32,
                        output32Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(deviceOutput64, output64Bytes, hostOutput64,
                        output64Bytes, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchSimtKeepResumeRegisterSwapKernel(deviceOutput32, deviceOutput64,
                                         stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(hostOutput32, output32Bytes, deviceOutput32,
                        output32Bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(hostOutput64, output64Bytes, deviceOutput64,
                        output64Bytes, ACL_MEMCPY_DEVICE_TO_HOST));

  output32File = std::fopen("./output_i32.bin", "wb");
  if (!output32File || std::fwrite(hostOutput32, 1, output32Bytes,
                                   output32File) != output32Bytes) {
    std::fprintf(stderr, "[ERROR] failed to write output_i32.bin\n");
    rc = 1;
    goto cleanup;
  }
  std::fclose(output32File);
  output32File = nullptr;
  output64File = std::fopen("./output_i64.bin", "wb");
  if (!output64File || std::fwrite(hostOutput64, 1, output64Bytes,
                                   output64File) != output64Bytes) {
    std::fprintf(stderr, "[ERROR] failed to write output_i64.bin\n");
    rc = 1;
    goto cleanup;
  }
  std::fclose(output64File);
  output64File = nullptr;

  for (int32_t tid = 0; tid < static_cast<int32_t>(kThreads); ++tid) {
    const int32_t actual32_0 = hostOutput32[2 * tid];
    const int32_t actual32_1 = hostOutput32[2 * tid + 1];
    const int64_t actual64_0 = hostOutput64[2 * tid];
    const int64_t actual64_1 = hostOutput64[2 * tid + 1];
    const int32_t expected32_0 = tid + 200;
    const int32_t expected32_1 = tid + 100;
    const int64_t expected64_0 = kBase64B + tid;
    const int64_t expected64_1 = kBase64A + tid;
    if (actual32_0 != expected32_0 || actual32_1 != expected32_1 ||
        actual64_0 != expected64_0 || actual64_1 != expected64_1) {
      std::fprintf(stderr,
                   "[ERROR] lane %d: got i32=[%d, %d], i64=[%lld, %lld]\n", tid,
                   actual32_0, actual32_1, static_cast<long long>(actual64_0),
                   static_cast<long long>(actual64_1));
      rc = 2;
      goto cleanup;
    }
  }
  std::printf("[PASS] all %zu lanes preserved the i32 and i64 swaps\n",
              kThreads);

cleanup:
  if (output64File)
    std::fclose(output64File);
  if (output32File)
    std::fclose(output32File);
  if (deviceOutput64)
    aclrtFree(deviceOutput64);
  if (deviceOutput32)
    aclrtFree(deviceOutput32);
  if (hostOutput64)
    aclrtFreeHost(hostOutput64);
  if (hostOutput32)
    aclrtFreeHost(hostOutput32);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInitialized)
    aclFinalize();
  return rc;
}
