// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Merged vdiv host runner.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { \
  const aclError _ret = (expr); \
  if (_ret != ACL_SUCCESS) { std::fprintf(stderr,"[ERROR] %s:%d acle=%d\n",#expr,__LINE__,(int)_ret); rc=1; goto cleanup; } \
} while(0)

#define FCK(expr,path) do { if(!(expr)){std::fprintf(stderr,"[ERROR] file:%s\n",path);rc=1;goto cleanup;} } while(0)



void LaunchVdivDeepMerged(float * p0, float * p1, float * p2, uint16_t * p3, uint16_t * p4, uint16_t * p5, float * p6, float * p7, float * p8, float * p9, float * p10, float * p11, void *stream);
int main() {
  constexpr size_t SZ_f32 = 4096;
  constexpr size_t SZ_f16 = 2048;
  constexpr size_t SZ_f32_exceptional = 4096;
  constexpr size_t SZ_tail = 4096;

  float *h_f32_v1=nullptr, *d_f32_v1=nullptr;
  float *h_f32_v2=nullptr, *d_f32_v2=nullptr;
  float *h_f32_v3=nullptr, *d_f32_v3=nullptr;
  uint16_t *h_f16_v1=nullptr, *d_f16_v1=nullptr;
  uint16_t *h_f16_v2=nullptr, *d_f16_v2=nullptr;
  uint16_t *h_f16_v3=nullptr, *d_f16_v3=nullptr;
  float *h_f32_exceptional_v1=nullptr, *d_f32_exceptional_v1=nullptr;
  float *h_f32_exceptional_v2=nullptr, *d_f32_exceptional_v2=nullptr;
  float *h_f32_exceptional_v3=nullptr, *d_f32_exceptional_v3=nullptr;
  float *h_tail_v1=nullptr, *d_tail_v1=nullptr;
  float *h_tail_v2=nullptr, *d_tail_v2=nullptr;
  float *h_tail_v3=nullptr, *d_tail_v3=nullptr;

  int rc=0; bool aclInited=false,deviceSet=false; int deviceId=0; aclrtStream stream=nullptr; size_t fsz=0;
  ACL_CHECK(aclInit(nullptr)); aclInited=true;
  if(const char*e=std::getenv("ACL_DEVICE_ID")) deviceId=std::atoi(e);
  ACL_CHECK(aclrtSetDevice(deviceId)); deviceSet=true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v1,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v2,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v3,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_f16_v1,SZ_f16));
  ACL_CHECK(aclrtMallocHost((void**)&h_f16_v2,SZ_f16));
  ACL_CHECK(aclrtMallocHost((void**)&h_f16_v3,SZ_f16));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_exceptional_v1,SZ_f32_exceptional));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_exceptional_v2,SZ_f32_exceptional));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_exceptional_v3,SZ_f32_exceptional));
  ACL_CHECK(aclrtMallocHost((void**)&h_tail_v1,SZ_tail));
  ACL_CHECK(aclrtMallocHost((void**)&h_tail_v2,SZ_tail));
  ACL_CHECK(aclrtMallocHost((void**)&h_tail_v3,SZ_tail));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v1,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v2,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v3,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f16_v1,SZ_f16,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f16_v2,SZ_f16,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f16_v3,SZ_f16,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_exceptional_v1,SZ_f32_exceptional,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_exceptional_v2,SZ_f32_exceptional,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_exceptional_v3,SZ_f32_exceptional,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_tail_v1,SZ_tail,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_tail_v2,SZ_tail,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_tail_v3,SZ_tail,ACL_MEM_MALLOC_HUGE_FIRST));
  fsz=SZ_f32; FCK(ReadFile("v1.bin",fsz,h_f32_v1,SZ_f32)&&fsz==SZ_f32,"v1.bin");
  fsz=SZ_f32; FCK(ReadFile("v2.bin",fsz,h_f32_v2,SZ_f32)&&fsz==SZ_f32,"v2.bin");
  fsz=SZ_f32; FCK(ReadFile("v3.bin",fsz,h_f32_v3,SZ_f32)&&fsz==SZ_f32,"v3.bin");
  fsz=SZ_f16; FCK(ReadFile("v1_f16.bin",fsz,h_f16_v1,SZ_f16)&&fsz==SZ_f16,"v1_f16.bin");
  fsz=SZ_f16; FCK(ReadFile("v2_f16.bin",fsz,h_f16_v2,SZ_f16)&&fsz==SZ_f16,"v2_f16.bin");
  fsz=SZ_f16; FCK(ReadFile("v3_f16.bin",fsz,h_f16_v3,SZ_f16)&&fsz==SZ_f16,"v3_f16.bin");
  fsz=SZ_f32_exceptional; FCK(ReadFile("v1_f32_exceptional.bin",fsz,h_f32_exceptional_v1,SZ_f32_exceptional)&&fsz==SZ_f32_exceptional,"v1_f32_exceptional.bin");
  fsz=SZ_f32_exceptional; FCK(ReadFile("v2_f32_exceptional.bin",fsz,h_f32_exceptional_v2,SZ_f32_exceptional)&&fsz==SZ_f32_exceptional,"v2_f32_exceptional.bin");
  fsz=SZ_f32_exceptional; FCK(ReadFile("v3_f32_exceptional.bin",fsz,h_f32_exceptional_v3,SZ_f32_exceptional)&&fsz==SZ_f32_exceptional,"v3_f32_exceptional.bin");
  fsz=SZ_tail; FCK(ReadFile("v1_tail.bin",fsz,h_tail_v1,SZ_tail)&&fsz==SZ_tail,"v1_tail.bin");
  fsz=SZ_tail; FCK(ReadFile("v2_tail.bin",fsz,h_tail_v2,SZ_tail)&&fsz==SZ_tail,"v2_tail.bin");
  fsz=SZ_tail; FCK(ReadFile("v3_tail.bin",fsz,h_tail_v3,SZ_tail)&&fsz==SZ_tail,"v3_tail.bin");
  ACL_CHECK(aclrtMemcpy(d_f32_v1,SZ_f32,h_f32_v1,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v2,SZ_f32,h_f32_v2,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v3,SZ_f32,h_f32_v3,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v1,SZ_f16,h_f16_v1,SZ_f16,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v2,SZ_f16,h_f16_v2,SZ_f16,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v3,SZ_f16,h_f16_v3,SZ_f16,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_exceptional_v1,SZ_f32_exceptional,h_f32_exceptional_v1,SZ_f32_exceptional,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_exceptional_v2,SZ_f32_exceptional,h_f32_exceptional_v2,SZ_f32_exceptional,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_exceptional_v3,SZ_f32_exceptional,h_f32_exceptional_v3,SZ_f32_exceptional,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_tail_v1,SZ_tail,h_tail_v1,SZ_tail,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_tail_v2,SZ_tail,h_tail_v2,SZ_tail,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_tail_v3,SZ_tail,h_tail_v3,SZ_tail,ACL_MEMCPY_HOST_TO_DEVICE));
    LaunchVdivDeepMerged(
      d_f32_v1,
      d_f32_v2,
      d_f32_v3,
      d_f16_v1,
      d_f16_v2,
      d_f16_v3,
      d_f32_exceptional_v1,
      d_f32_exceptional_v2,
      d_f32_exceptional_v3,
      d_tail_v1,
      d_tail_v2,
      d_tail_v3,
      stream
  );
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(h_f32_v3,SZ_f32,d_f32_v3,SZ_f32,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_f16_v3,SZ_f16,d_f16_v3,SZ_f16,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_f32_exceptional_v3,SZ_f32_exceptional,d_f32_exceptional_v3,SZ_f32_exceptional,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_tail_v3,SZ_tail,d_tail_v3,SZ_tail,ACL_MEMCPY_DEVICE_TO_HOST));
  FCK(WriteFile("v3.bin",h_f32_v3,SZ_f32),"v3.bin");
  FCK(WriteFile("v3_f16.bin",h_f16_v3,SZ_f16),"v3_f16.bin");
  FCK(WriteFile("v3_f32_exceptional.bin",h_f32_exceptional_v3,SZ_f32_exceptional),"v3_f32_exceptional.bin");
  FCK(WriteFile("v3_tail.bin",h_tail_v3,SZ_tail),"v3_tail.bin");

cleanup:
  aclrtFree(d_f32_v1);
  aclrtFree(d_f32_v2);
  aclrtFree(d_f32_v3);
  aclrtFree(d_f16_v1);
  aclrtFree(d_f16_v2);
  aclrtFree(d_f16_v3);
  aclrtFree(d_f32_exceptional_v1);
  aclrtFree(d_f32_exceptional_v2);
  aclrtFree(d_f32_exceptional_v3);
  aclrtFree(d_tail_v1);
  aclrtFree(d_tail_v2);
  aclrtFree(d_tail_v3);
  aclrtFreeHost(h_f32_v1);
  aclrtFreeHost(h_f32_v2);
  aclrtFreeHost(h_f32_v3);
  aclrtFreeHost(h_f16_v1);
  aclrtFreeHost(h_f16_v2);
  aclrtFreeHost(h_f16_v3);
  aclrtFreeHost(h_f32_exceptional_v1);
  aclrtFreeHost(h_f32_exceptional_v2);
  aclrtFreeHost(h_f32_exceptional_v3);
  aclrtFreeHost(h_tail_v1);
  aclrtFreeHost(h_tail_v2);
  aclrtFreeHost(h_tail_v3);
  if(stream) aclrtDestroyStream(stream);
  if(deviceSet) aclrtResetDevice(deviceId);
  if(aclInited) aclFinalize();
  return rc;
}
