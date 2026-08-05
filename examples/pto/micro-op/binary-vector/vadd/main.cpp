// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Merged vadd host runner: 9 variants run sequentially.
// Each variant uses uniquely suffixed .bin files to avoid collisions.

#include "test_common.h"
#include "acl/acl.h"
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

// Launch wrappers

void LaunchVaddDeepMerged(float * p0, float * p1, float * p2, uint16_t * p3, uint16_t * p4, uint16_t * p5, uint16_t * p6, uint16_t * p7, uint16_t * p8, float * p9, float * p10, float * p11, int16_t * p12, int16_t * p13, int16_t * p14, int16_t * p15, int16_t * p16, int16_t * p17, uint16_t * p18, uint16_t * p19, uint16_t * p20, uint16_t * p21, uint16_t * p22, uint16_t * p23, float * p24, float * p25, float * p26, void *stream);
int main() {
  // ----- sizes (element counts * sizeof per variant) -----
  constexpr size_t ELEM_F32   = 1024;
  constexpr size_t ELEM_F16   = 1024;
  constexpr size_t ELEM_BF16  = 1024;
  constexpr size_t ELEM_X     = 1024;
  constexpr size_t ELEM_I16S  = 1024;
  constexpr size_t ELEM_I16U  = 1024;
  constexpr size_t ELEM_TAIL  = 1024;

  constexpr size_t SZ_F32   = ELEM_F32   * sizeof(float);
  constexpr size_t SZ_F16   = ELEM_F16   * sizeof(uint16_t);
  constexpr size_t SZ_BF16  = ELEM_BF16  * sizeof(uint16_t);
  constexpr size_t SZ_X     = ELEM_X     * sizeof(float);
  constexpr size_t SZ_I16S  = ELEM_I16S  * sizeof(int16_t);
  constexpr size_t SZ_I16U  = ELEM_I16U  * sizeof(uint16_t);
  constexpr size_t SZ_TAIL  = ELEM_TAIL  * sizeof(float);

  // ----- host/device pointers (3 buffers per variant) -----
  float    *h_f32_v1 = nullptr, *h_f32_v2 = nullptr, *h_f32_v3 = nullptr;
  float    *d_f32_v1 = nullptr, *d_f32_v2 = nullptr, *d_f32_v3 = nullptr;

  uint16_t *h_f16_v1 = nullptr, *h_f16_v2 = nullptr, *h_f16_v3 = nullptr;
  uint16_t *d_f16_v1 = nullptr, *d_f16_v2 = nullptr, *d_f16_v3 = nullptr;

  uint16_t *h_bf16_v1 = nullptr, *h_bf16_v2 = nullptr, *h_bf16_v3 = nullptr;
  uint16_t *d_bf16_v1 = nullptr, *d_bf16_v2 = nullptr, *d_bf16_v3 = nullptr;

  float    *h_x_v1 = nullptr, *h_x_v2 = nullptr, *h_x_v3 = nullptr;
  float    *d_x_v1 = nullptr, *d_x_v2 = nullptr, *d_x_v3 = nullptr;

  int16_t  *h_i16s_v1 = nullptr, *h_i16s_v2 = nullptr, *h_i16s_v3 = nullptr;
  int16_t  *d_i16s_v1 = nullptr, *d_i16s_v2 = nullptr, *d_i16s_v3 = nullptr;

  int16_t  *h_i16so_v1 = nullptr, *h_i16so_v2 = nullptr, *h_i16so_v3 = nullptr;
  int16_t  *d_i16so_v1 = nullptr, *d_i16so_v2 = nullptr, *d_i16so_v3 = nullptr;

  uint16_t *h_i16u_v1 = nullptr, *h_i16u_v2 = nullptr, *h_i16u_v3 = nullptr;
  uint16_t *d_i16u_v1 = nullptr, *d_i16u_v2 = nullptr, *d_i16u_v3 = nullptr;

  uint16_t *h_i16uo_v1 = nullptr, *h_i16uo_v2 = nullptr, *h_i16uo_v3 = nullptr;
  uint16_t *d_i16uo_v1 = nullptr, *d_i16uo_v2 = nullptr, *d_i16uo_v3 = nullptr;

  float    *h_tail_v1 = nullptr, *h_tail_v2 = nullptr, *h_tail_v3 = nullptr;
  float    *d_tail_v1 = nullptr, *d_tail_v2 = nullptr, *d_tail_v3 = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t fsize = 0;

  // ----- init ACL -----
  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  // ----- host malloc (all variants) -----
  ACL_CHECK(aclrtMallocHost((void **)&h_f32_v1,  SZ_F32));
  ACL_CHECK(aclrtMallocHost((void **)&h_f32_v2,  SZ_F32));
  ACL_CHECK(aclrtMallocHost((void **)&h_f32_v3,  SZ_F32));
  ACL_CHECK(aclrtMallocHost((void **)&h_f16_v1,  SZ_F16));
  ACL_CHECK(aclrtMallocHost((void **)&h_f16_v2,  SZ_F16));
  ACL_CHECK(aclrtMallocHost((void **)&h_f16_v3,  SZ_F16));
  ACL_CHECK(aclrtMallocHost((void **)&h_bf16_v1, SZ_BF16));
  ACL_CHECK(aclrtMallocHost((void **)&h_bf16_v2, SZ_BF16));
  ACL_CHECK(aclrtMallocHost((void **)&h_bf16_v3, SZ_BF16));
  ACL_CHECK(aclrtMallocHost((void **)&h_x_v1,    SZ_X));
  ACL_CHECK(aclrtMallocHost((void **)&h_x_v2,    SZ_X));
  ACL_CHECK(aclrtMallocHost((void **)&h_x_v3,    SZ_X));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16s_v1, SZ_I16S));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16s_v2, SZ_I16S));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16s_v3, SZ_I16S));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16so_v1,SZ_I16S));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16so_v2,SZ_I16S));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16so_v3,SZ_I16S));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16u_v1, SZ_I16U));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16u_v2, SZ_I16U));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16u_v3, SZ_I16U));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16uo_v1,SZ_I16U));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16uo_v2,SZ_I16U));
  ACL_CHECK(aclrtMallocHost((void **)&h_i16uo_v3,SZ_I16U));
  ACL_CHECK(aclrtMallocHost((void **)&h_tail_v1, SZ_TAIL));
  ACL_CHECK(aclrtMallocHost((void **)&h_tail_v2, SZ_TAIL));
  ACL_CHECK(aclrtMallocHost((void **)&h_tail_v3, SZ_TAIL));

  // ----- device malloc (all variants) -----
  ACL_CHECK(aclrtMalloc((void **)&d_f32_v1,  SZ_F32,  ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_f32_v2,  SZ_F32,  ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_f32_v3,  SZ_F32,  ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_f16_v1,  SZ_F16,  ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_f16_v2,  SZ_F16,  ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_f16_v3,  SZ_F16,  ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_bf16_v1, SZ_BF16, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_bf16_v2, SZ_BF16, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_bf16_v3, SZ_BF16, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_x_v1,    SZ_X,    ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_x_v2,    SZ_X,    ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_x_v3,    SZ_X,    ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16s_v1, SZ_I16S, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16s_v2, SZ_I16S, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16s_v3, SZ_I16S, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16so_v1,SZ_I16S, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16so_v2,SZ_I16S, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16so_v3,SZ_I16S, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16u_v1, SZ_I16U, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16u_v2, SZ_I16U, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16u_v3, SZ_I16U, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16uo_v1,SZ_I16U, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16uo_v2,SZ_I16U, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_i16uo_v3,SZ_I16U, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_tail_v1, SZ_TAIL, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_tail_v2, SZ_TAIL, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_tail_v3, SZ_TAIL, ACL_MEM_MALLOC_HUGE_FIRST));

  // ----- read inputs (all variants) -----
  fsize = SZ_F32;  FILE_CHECK(ReadFile("./v1.bin", fsize, h_f32_v1, SZ_F32) && fsize == SZ_F32, "./v1.bin");
  fsize = SZ_F32;  FILE_CHECK(ReadFile("./v2.bin", fsize, h_f32_v2, SZ_F32) && fsize == SZ_F32, "./v2.bin");
  fsize = SZ_F32;  FILE_CHECK(ReadFile("./v3.bin", fsize, h_f32_v3, SZ_F32) && fsize == SZ_F32, "./v3.bin");

  fsize = SZ_F16;  FILE_CHECK(ReadFile("./v1_f16.bin", fsize, h_f16_v1, SZ_F16) && fsize == SZ_F16, "./v1_f16.bin");
  fsize = SZ_F16;  FILE_CHECK(ReadFile("./v2_f16.bin", fsize, h_f16_v2, SZ_F16) && fsize == SZ_F16, "./v2_f16.bin");
  fsize = SZ_F16;  FILE_CHECK(ReadFile("./v3_f16.bin", fsize, h_f16_v3, SZ_F16) && fsize == SZ_F16, "./v3_f16.bin");

  fsize = SZ_BF16; FILE_CHECK(ReadFile("./v1_bf16.bin", fsize, h_bf16_v1, SZ_BF16) && fsize == SZ_BF16, "./v1_bf16.bin");
  fsize = SZ_BF16; FILE_CHECK(ReadFile("./v2_bf16.bin", fsize, h_bf16_v2, SZ_BF16) && fsize == SZ_BF16, "./v2_bf16.bin");
  fsize = SZ_BF16; FILE_CHECK(ReadFile("./v3_bf16.bin", fsize, h_bf16_v3, SZ_BF16) && fsize == SZ_BF16, "./v3_bf16.bin");

  fsize = SZ_X;    FILE_CHECK(ReadFile("./v1_x.bin", fsize, h_x_v1, SZ_X) && fsize == SZ_X, "./v1_x.bin");
  fsize = SZ_X;    FILE_CHECK(ReadFile("./v2_x.bin", fsize, h_x_v2, SZ_X) && fsize == SZ_X, "./v2_x.bin");
  fsize = SZ_X;    FILE_CHECK(ReadFile("./v3_x.bin", fsize, h_x_v3, SZ_X) && fsize == SZ_X, "./v3_x.bin");

  fsize = SZ_I16S; FILE_CHECK(ReadFile("./v1_i16s.bin", fsize, h_i16s_v1, SZ_I16S) && fsize == SZ_I16S, "./v1_i16s.bin");
  fsize = SZ_I16S; FILE_CHECK(ReadFile("./v2_i16s.bin", fsize, h_i16s_v2, SZ_I16S) && fsize == SZ_I16S, "./v2_i16s.bin");
  fsize = SZ_I16S; FILE_CHECK(ReadFile("./v3_i16s.bin", fsize, h_i16s_v3, SZ_I16S) && fsize == SZ_I16S, "./v3_i16s.bin");

  fsize = SZ_I16S; FILE_CHECK(ReadFile("./v1_i16s_ov.bin", fsize, h_i16so_v1, SZ_I16S) && fsize == SZ_I16S, "./v1_i16s_ov.bin");
  fsize = SZ_I16S; FILE_CHECK(ReadFile("./v2_i16s_ov.bin", fsize, h_i16so_v2, SZ_I16S) && fsize == SZ_I16S, "./v2_i16s_ov.bin");
  fsize = SZ_I16S; FILE_CHECK(ReadFile("./v3_i16s_ov.bin", fsize, h_i16so_v3, SZ_I16S) && fsize == SZ_I16S, "./v3_i16s_ov.bin");

  fsize = SZ_I16U; FILE_CHECK(ReadFile("./v1_i16u.bin", fsize, h_i16u_v1, SZ_I16U) && fsize == SZ_I16U, "./v1_i16u.bin");
  fsize = SZ_I16U; FILE_CHECK(ReadFile("./v2_i16u.bin", fsize, h_i16u_v2, SZ_I16U) && fsize == SZ_I16U, "./v2_i16u.bin");
  fsize = SZ_I16U; FILE_CHECK(ReadFile("./v3_i16u.bin", fsize, h_i16u_v3, SZ_I16U) && fsize == SZ_I16U, "./v3_i16u.bin");

  fsize = SZ_I16U; FILE_CHECK(ReadFile("./v1_i16u_ov.bin", fsize, h_i16uo_v1, SZ_I16U) && fsize == SZ_I16U, "./v1_i16u_ov.bin");
  fsize = SZ_I16U; FILE_CHECK(ReadFile("./v2_i16u_ov.bin", fsize, h_i16uo_v2, SZ_I16U) && fsize == SZ_I16U, "./v2_i16u_ov.bin");
  fsize = SZ_I16U; FILE_CHECK(ReadFile("./v3_i16u_ov.bin", fsize, h_i16uo_v3, SZ_I16U) && fsize == SZ_I16U, "./v3_i16u_ov.bin");

  fsize = SZ_TAIL; FILE_CHECK(ReadFile("./v1_tail.bin", fsize, h_tail_v1, SZ_TAIL) && fsize == SZ_TAIL, "./v1_tail.bin");
  fsize = SZ_TAIL; FILE_CHECK(ReadFile("./v2_tail.bin", fsize, h_tail_v2, SZ_TAIL) && fsize == SZ_TAIL, "./v2_tail.bin");
  fsize = SZ_TAIL; FILE_CHECK(ReadFile("./v3_tail.bin", fsize, h_tail_v3, SZ_TAIL) && fsize == SZ_TAIL, "./v3_tail.bin");

  // ----- H2D copies (all variants) -----
  ACL_CHECK(aclrtMemcpy(d_f32_v1,  SZ_F32,  h_f32_v1,  SZ_F32,  ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v2,  SZ_F32,  h_f32_v2,  SZ_F32,  ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v3,  SZ_F32,  h_f32_v3,  SZ_F32,  ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v1,  SZ_F16,  h_f16_v1,  SZ_F16,  ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v2,  SZ_F16,  h_f16_v2,  SZ_F16,  ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v3,  SZ_F16,  h_f16_v3,  SZ_F16,  ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_bf16_v1, SZ_BF16, h_bf16_v1, SZ_BF16, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_bf16_v2, SZ_BF16, h_bf16_v2, SZ_BF16, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_bf16_v3, SZ_BF16, h_bf16_v3, SZ_BF16, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_x_v1,    SZ_X,    h_x_v1,    SZ_X,    ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_x_v2,    SZ_X,    h_x_v2,    SZ_X,    ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_x_v3,    SZ_X,    h_x_v3,    SZ_X,    ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16s_v1, SZ_I16S, h_i16s_v1, SZ_I16S, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16s_v2, SZ_I16S, h_i16s_v2, SZ_I16S, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16s_v3, SZ_I16S, h_i16s_v3, SZ_I16S, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16so_v1,SZ_I16S, h_i16so_v1,SZ_I16S, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16so_v2,SZ_I16S, h_i16so_v2,SZ_I16S, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16so_v3,SZ_I16S, h_i16so_v3,SZ_I16S, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16u_v1, SZ_I16U, h_i16u_v1, SZ_I16U, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16u_v2, SZ_I16U, h_i16u_v2, SZ_I16U, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16u_v3, SZ_I16U, h_i16u_v3, SZ_I16U, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16uo_v1,SZ_I16U, h_i16uo_v1,SZ_I16U, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16uo_v2,SZ_I16U, h_i16uo_v2,SZ_I16U, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i16uo_v3,SZ_I16U, h_i16uo_v3,SZ_I16U, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_tail_v1, SZ_TAIL, h_tail_v1, SZ_TAIL, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_tail_v2, SZ_TAIL, h_tail_v2, SZ_TAIL, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_tail_v3, SZ_TAIL, h_tail_v3, SZ_TAIL, ACL_MEMCPY_HOST_TO_DEVICE));

  // ----- launch all 9 kernels -----
    LaunchVaddDeepMerged(
      d_f32_v1,
      d_f32_v2,
      d_f32_v3,
      d_f16_v1,
      d_f16_v2,
      d_f16_v3,
      d_bf16_v1,
      d_bf16_v2,
      d_bf16_v3,
      d_x_v1,
      d_x_v2,
      d_x_v3,
      d_i16s_v1,
      d_i16s_v2,
      d_i16s_v3,
      d_i16so_v1,
      d_i16so_v2,
      d_i16so_v3,
      d_i16u_v1,
      d_i16u_v2,
      d_i16u_v3,
      d_i16uo_v1,
      d_i16uo_v2,
      d_i16uo_v3,
      d_tail_v1,
      d_tail_v2,
      d_tail_v3,
      stream
  );
  ACL_CHECK(aclrtSynchronizeStream(stream));

  // ----- D2H copies (outputs) -----
  ACL_CHECK(aclrtMemcpy(h_f32_v3,  SZ_F32,  d_f32_v3,  SZ_F32,  ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_f16_v3,  SZ_F16,  d_f16_v3,  SZ_F16,  ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_bf16_v3, SZ_BF16, d_bf16_v3, SZ_BF16, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_x_v3,    SZ_X,    d_x_v3,    SZ_X,    ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_i16s_v3, SZ_I16S, d_i16s_v3, SZ_I16S, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_i16so_v3,SZ_I16S, d_i16so_v3,SZ_I16S, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_i16u_v3, SZ_I16U, d_i16u_v3, SZ_I16U, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_i16uo_v3,SZ_I16U, d_i16uo_v3,SZ_I16U, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_tail_v3, SZ_TAIL, d_tail_v3, SZ_TAIL, ACL_MEMCPY_DEVICE_TO_HOST));

  // ----- write outputs -----
  FILE_CHECK(WriteFile("./v3.bin",      h_f32_v3,  SZ_F32),  "./v3.bin");
  FILE_CHECK(WriteFile("./v3_f16.bin",  h_f16_v3,  SZ_F16),  "./v3_f16.bin");
  FILE_CHECK(WriteFile("./v3_bf16.bin", h_bf16_v3, SZ_BF16), "./v3_bf16.bin");
  FILE_CHECK(WriteFile("./v3_x.bin",    h_x_v3,    SZ_X),    "./v3_x.bin");
  FILE_CHECK(WriteFile("./v3_i16s.bin", h_i16s_v3, SZ_I16S), "./v3_i16s.bin");
  FILE_CHECK(WriteFile("./v3_i16s_ov.bin",h_i16so_v3,SZ_I16S),"./v3_i16s_ov.bin");
  FILE_CHECK(WriteFile("./v3_i16u.bin", h_i16u_v3, SZ_I16U), "./v3_i16u.bin");
  FILE_CHECK(WriteFile("./v3_i16u_ov.bin",h_i16uo_v3,SZ_I16U),"./v3_i16u_ov.bin");
  FILE_CHECK(WriteFile("./v3_tail.bin", h_tail_v3, SZ_TAIL), "./v3_tail.bin");

cleanup:
  aclrtFree(d_f32_v1);  aclrtFree(d_f32_v2);  aclrtFree(d_f32_v3);
  aclrtFree(d_f16_v1);  aclrtFree(d_f16_v2);  aclrtFree(d_f16_v3);
  aclrtFree(d_bf16_v1); aclrtFree(d_bf16_v2); aclrtFree(d_bf16_v3);
  aclrtFree(d_x_v1);    aclrtFree(d_x_v2);    aclrtFree(d_x_v3);
  aclrtFree(d_i16s_v1); aclrtFree(d_i16s_v2); aclrtFree(d_i16s_v3);
  aclrtFree(d_i16so_v1);aclrtFree(d_i16so_v2);aclrtFree(d_i16so_v3);
  aclrtFree(d_i16u_v1); aclrtFree(d_i16u_v2); aclrtFree(d_i16u_v3);
  aclrtFree(d_i16uo_v1);aclrtFree(d_i16uo_v2);aclrtFree(d_i16uo_v3);
  aclrtFree(d_tail_v1); aclrtFree(d_tail_v2); aclrtFree(d_tail_v3);
  aclrtFreeHost(h_f32_v1);  aclrtFreeHost(h_f32_v2);  aclrtFreeHost(h_f32_v3);
  aclrtFreeHost(h_f16_v1);  aclrtFreeHost(h_f16_v2);  aclrtFreeHost(h_f16_v3);
  aclrtFreeHost(h_bf16_v1); aclrtFreeHost(h_bf16_v2); aclrtFreeHost(h_bf16_v3);
  aclrtFreeHost(h_x_v1);    aclrtFreeHost(h_x_v2);    aclrtFreeHost(h_x_v3);
  aclrtFreeHost(h_i16s_v1); aclrtFreeHost(h_i16s_v2); aclrtFreeHost(h_i16s_v3);
  aclrtFreeHost(h_i16so_v1);aclrtFreeHost(h_i16so_v2);aclrtFreeHost(h_i16so_v3);
  aclrtFreeHost(h_i16u_v1); aclrtFreeHost(h_i16u_v2); aclrtFreeHost(h_i16u_v3);
  aclrtFreeHost(h_i16uo_v1);aclrtFreeHost(h_i16uo_v2);aclrtFreeHost(h_i16uo_v3);
  aclrtFreeHost(h_tail_v1); aclrtFreeHost(h_tail_v2); aclrtFreeHost(h_tail_v3);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
