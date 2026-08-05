// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Merged vshl test case.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
using namespace PtoTestCommon;

#define ACL_CHECK(expr) do {   const aclError _r=(expr);   if(_r!=ACL_SUCCESS){std::fprintf(stderr,"[ERROR] %s:%d acle=%d\n",#expr,__LINE__,(int)_r);rc=1;goto cleanup;} }while(0)
#define FCK(expr,path) do{if(!(expr)){std::fprintf(stderr,"[ERROR] file:%s\n",path);rc=1;goto cleanup;}}while(0)



void LaunchVshlDeepMerged(uint16_t * p0, uint16_t * p1, uint16_t * p2, uint32_t * p3, uint32_t * p4, uint32_t * p5, uint16_t * p6, uint16_t * p7, uint16_t * p8, void *stream);
int main() {
  constexpr size_t SZ_f32 = 2048;
  constexpr size_t SZ_i32_unsigned = 4096;
  constexpr size_t SZ_shift_boundary = 2048;

  uint16_t *h_f32_v1=nullptr, *d_f32_v1=nullptr;
  uint16_t *h_f32_v2=nullptr, *d_f32_v2=nullptr;
  uint16_t *h_f32_v3=nullptr, *d_f32_v3=nullptr;
  uint32_t *h_i32_unsigned_v1=nullptr, *d_i32_unsigned_v1=nullptr;
  uint32_t *h_i32_unsigned_v2=nullptr, *d_i32_unsigned_v2=nullptr;
  uint32_t *h_i32_unsigned_v3=nullptr, *d_i32_unsigned_v3=nullptr;
  uint16_t *h_shift_boundary_v1=nullptr, *d_shift_boundary_v1=nullptr;
  uint16_t *h_shift_boundary_v2=nullptr, *d_shift_boundary_v2=nullptr;
  uint16_t *h_shift_boundary_v3=nullptr, *d_shift_boundary_v3=nullptr;
  int rc=0; bool aclInited=false,deviceSet=false; int deviceId=0; aclrtStream stream=nullptr; size_t fsz=0;
  ACL_CHECK(aclInit(nullptr)); aclInited=true;
  if(const char*e=std::getenv("ACL_DEVICE_ID")) deviceId=std::atoi(e);
  ACL_CHECK(aclrtSetDevice(deviceId)); deviceSet=true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v1,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v2,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v3,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_i32_unsigned_v1,SZ_i32_unsigned));
  ACL_CHECK(aclrtMallocHost((void**)&h_i32_unsigned_v2,SZ_i32_unsigned));
  ACL_CHECK(aclrtMallocHost((void**)&h_i32_unsigned_v3,SZ_i32_unsigned));
  ACL_CHECK(aclrtMallocHost((void**)&h_shift_boundary_v1,SZ_shift_boundary));
  ACL_CHECK(aclrtMallocHost((void**)&h_shift_boundary_v2,SZ_shift_boundary));
  ACL_CHECK(aclrtMallocHost((void**)&h_shift_boundary_v3,SZ_shift_boundary));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v1,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v2,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v3,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_i32_unsigned_v1,SZ_i32_unsigned,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_i32_unsigned_v2,SZ_i32_unsigned,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_i32_unsigned_v3,SZ_i32_unsigned,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_shift_boundary_v1,SZ_shift_boundary,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_shift_boundary_v2,SZ_shift_boundary,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_shift_boundary_v3,SZ_shift_boundary,ACL_MEM_MALLOC_HUGE_FIRST));
  fsz=SZ_f32; FCK(ReadFile("v1.bin",fsz,h_f32_v1,SZ_f32)&&fsz==SZ_f32,"v1.bin");
  fsz=SZ_f32; FCK(ReadFile("v2.bin",fsz,h_f32_v2,SZ_f32)&&fsz==SZ_f32,"v2.bin");
  fsz=SZ_f32; FCK(ReadFile("v3.bin",fsz,h_f32_v3,SZ_f32)&&fsz==SZ_f32,"v3.bin");
  fsz=SZ_i32_unsigned; FCK(ReadFile("v1_i32_unsigned.bin",fsz,h_i32_unsigned_v1,SZ_i32_unsigned)&&fsz==SZ_i32_unsigned,"v1_i32_unsigned.bin");
  fsz=SZ_i32_unsigned; FCK(ReadFile("v2_i32_unsigned.bin",fsz,h_i32_unsigned_v2,SZ_i32_unsigned)&&fsz==SZ_i32_unsigned,"v2_i32_unsigned.bin");
  fsz=SZ_i32_unsigned; FCK(ReadFile("v3_i32_unsigned.bin",fsz,h_i32_unsigned_v3,SZ_i32_unsigned)&&fsz==SZ_i32_unsigned,"v3_i32_unsigned.bin");
  fsz=SZ_shift_boundary; FCK(ReadFile("v1_shift_boundary.bin",fsz,h_shift_boundary_v1,SZ_shift_boundary)&&fsz==SZ_shift_boundary,"v1_shift_boundary.bin");
  fsz=SZ_shift_boundary; FCK(ReadFile("v2_shift_boundary.bin",fsz,h_shift_boundary_v2,SZ_shift_boundary)&&fsz==SZ_shift_boundary,"v2_shift_boundary.bin");
  fsz=SZ_shift_boundary; FCK(ReadFile("v3_shift_boundary.bin",fsz,h_shift_boundary_v3,SZ_shift_boundary)&&fsz==SZ_shift_boundary,"v3_shift_boundary.bin");
  ACL_CHECK(aclrtMemcpy(d_f32_v1,SZ_f32,h_f32_v1,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v2,SZ_f32,h_f32_v2,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v3,SZ_f32,h_f32_v3,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i32_unsigned_v1,SZ_i32_unsigned,h_i32_unsigned_v1,SZ_i32_unsigned,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i32_unsigned_v2,SZ_i32_unsigned,h_i32_unsigned_v2,SZ_i32_unsigned,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_i32_unsigned_v3,SZ_i32_unsigned,h_i32_unsigned_v3,SZ_i32_unsigned,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_shift_boundary_v1,SZ_shift_boundary,h_shift_boundary_v1,SZ_shift_boundary,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_shift_boundary_v2,SZ_shift_boundary,h_shift_boundary_v2,SZ_shift_boundary,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_shift_boundary_v3,SZ_shift_boundary,h_shift_boundary_v3,SZ_shift_boundary,ACL_MEMCPY_HOST_TO_DEVICE));
    LaunchVshlDeepMerged(
      d_f32_v1,
      d_f32_v2,
      d_f32_v3,
      d_i32_unsigned_v1,
      d_i32_unsigned_v2,
      d_i32_unsigned_v3,
      d_shift_boundary_v1,
      d_shift_boundary_v2,
      d_shift_boundary_v3,
      stream
  );
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(h_f32_v3,SZ_f32,d_f32_v3,SZ_f32,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_i32_unsigned_v3,SZ_i32_unsigned,d_i32_unsigned_v3,SZ_i32_unsigned,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_shift_boundary_v3,SZ_shift_boundary,d_shift_boundary_v3,SZ_shift_boundary,ACL_MEMCPY_DEVICE_TO_HOST));
  FCK(WriteFile("v3.bin",h_f32_v3,SZ_f32),"v3.bin");
  FCK(WriteFile("v3_i32_unsigned.bin",h_i32_unsigned_v3,SZ_i32_unsigned),"v3_i32_unsigned.bin");
  FCK(WriteFile("v3_shift_boundary.bin",h_shift_boundary_v3,SZ_shift_boundary),"v3_shift_boundary.bin");

cleanup:
  aclrtFree(d_f32_v1);
  aclrtFree(d_f32_v2);
  aclrtFree(d_f32_v3);
  aclrtFree(d_i32_unsigned_v1);
  aclrtFree(d_i32_unsigned_v2);
  aclrtFree(d_i32_unsigned_v3);
  aclrtFree(d_shift_boundary_v1);
  aclrtFree(d_shift_boundary_v2);
  aclrtFree(d_shift_boundary_v3);
  aclrtFreeHost(h_f32_v1);
  aclrtFreeHost(h_f32_v2);
  aclrtFreeHost(h_f32_v3);
  aclrtFreeHost(h_i32_unsigned_v1);
  aclrtFreeHost(h_i32_unsigned_v2);
  aclrtFreeHost(h_i32_unsigned_v3);
  aclrtFreeHost(h_shift_boundary_v1);
  aclrtFreeHost(h_shift_boundary_v2);
  aclrtFreeHost(h_shift_boundary_v3);
  if(stream) aclrtDestroyStream(stream);
  if(deviceSet) aclrtResetDevice(deviceId);
  if(aclInited) aclFinalize();
  return rc;
}
