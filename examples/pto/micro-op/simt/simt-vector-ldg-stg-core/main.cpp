// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { const aclError _ret = (expr); if (_ret != ACL_SUCCESS) { std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); rc = 1; goto cleanup; } } while (0)

// Arithmetic buffers: 12 vectors × 2 elements = 24 elements each
#define ARITH_ELEMS 24
// Copy buffers: 4 elements (2 input + 2 output), each element = 2 bytes
#define COPY_SIZE 8

void LaunchSimt_vector_ldg_stg_core_kernel(
    void *f32x2, void *f16x2, void *bf16x2, void *i16x2, void *i32x2,
    void *hif8x2, void *i8x2, void *fp8e4x2, void *fp8e5x2, void *stream);

struct Buffer {
  const char *name;
  void *host;
  void *device;
  size_t size;
  bool inited;
  Buffer() : name(nullptr), host(nullptr), device(nullptr), size(0), inited(false) {}
  void setup(const char *n, size_t s) { name = n; size = s; }
};

static int initBuffer(Buffer *b) {
  aclError ret;
  ret = aclrtMallocHost(&b->host, b->size);
  if (ret != ACL_SUCCESS) return 1;
  ret = aclrtMalloc((void **)&b->device, b->size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) return 1;
  ReadFile(b->name, b->size, b->host, b->size);
  ret = aclrtMemcpy(b->device, b->size, b->host, b->size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) return 1;
  b->inited = true;
  return 0;
}

static int readBack(Buffer *b) {
  aclError ret = aclrtMemcpy(b->host, b->size, b->device, b->size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) return 1;
  WriteFile(b->name, b->host, b->size);
  return 0;
}

static void freeBuffer(Buffer *b) {
  if (b->inited) {
    if (b->device) aclrtFree(b->device);
    if (b->host) aclrtFreeHost(b->host);
  }
}

int main() {
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  // Declare all buffers before any goto cleanup (C++ forbids jumping over initializers)
  Buffer bufF32x2, bufF16x2, bufBf16x2, bufI16x2, bufI32x2;
  Buffer bufHif8x2, bufI8x2, bufFp8e4x2, bufFp8e5x2;

  bufF32x2.setup("f32x2.bin", ARITH_ELEMS * sizeof(float));
  bufF16x2.setup("f16x2.bin", ARITH_ELEMS * sizeof(uint16_t));
  bufBf16x2.setup("bf16x2.bin", ARITH_ELEMS * sizeof(uint16_t));
  bufI16x2.setup("i16x2.bin", ARITH_ELEMS * sizeof(uint16_t));
  bufI32x2.setup("i32x2.bin", ARITH_ELEMS * sizeof(uint32_t));
  bufHif8x2.setup("hif8x2.bin", COPY_SIZE);
  bufI8x2.setup("i8x2.bin", COPY_SIZE);
  bufFp8e4x2.setup("fp8e4x2.bin", COPY_SIZE);
  bufFp8e5x2.setup("fp8e5x2.bin", COPY_SIZE);

  Buffer *allBufs[] = {
    &bufF32x2, &bufF16x2, &bufBf16x2, &bufI16x2, &bufI32x2,
    &bufHif8x2, &bufI8x2, &bufFp8e4x2, &bufFp8e5x2
  };

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  for (auto *b : allBufs) {
    if (initBuffer(b)) {
      std::fprintf(stderr, "[ERROR] initBuffer failed for %s\n", b->name);
      rc = 1; goto cleanup;
    }
  }

  LaunchSimt_vector_ldg_stg_core_kernel(
      bufF32x2.device, bufF16x2.device, bufBf16x2.device,
      bufI16x2.device, bufI32x2.device,
      bufHif8x2.device, bufI8x2.device, bufFp8e4x2.device, bufFp8e5x2.device,
      stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (auto *b : allBufs) {
    if (readBack(b)) {
      std::fprintf(stderr, "[ERROR] readBack failed for %s\n", b->name);
      rc = 1; goto cleanup;
    }
  }

cleanup:
  for (auto *b : allBufs) freeBuffer(b);
  if (stream)   aclrtDestroyStream(stream);
  if (deviceSet) aclrtResetDevice(deviceId);
  if (aclInited) aclFinalize();
  return rc;
}
