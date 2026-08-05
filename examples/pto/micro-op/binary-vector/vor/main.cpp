// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Merged vor test case.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
using namespace PtoTestCommon;

#define ACL_CHECK(expr) do {   const aclError _r=(expr);   if(_r!=ACL_SUCCESS){std::fprintf(stderr,"[ERROR] %s:%d acle=%d\n",#expr,__LINE__,(int)_r);rc=1;goto cleanup;} }while(0)
#define FCK(expr,path) do{if(!(expr)){std::fprintf(stderr,"[ERROR] file:%s\n",path);rc=1;goto cleanup;}}while(0)




void LaunchVorDeepMerged(uint16_t * p0, uint16_t * p1, uint16_t * p2, uint16_t * p3, uint16_t * p4, uint16_t * p5, uint16_t * p6, uint16_t * p7, uint16_t * p8, void *stream);
int main() {
  constexpr size_t SZ_f32 = 2048;
  constexpr size_t SZ_f16 = 2048;
  constexpr size_t SZ_mask_edge = 2048;

  uint16_t *h_f32_v1=nullptr, *d_f32_v1=nullptr;
  uint16_t *h_f32_v2=nullptr, *d_f32_v2=nullptr;
  uint16_t *h_f32_v3=nullptr, *d_f32_v3=nullptr;
  uint16_t *h_f16_v1=nullptr, *d_f16_v1=nullptr;
  uint16_t *h_f16_v2=nullptr, *d_f16_v2=nullptr;
  uint16_t *h_f16_v3=nullptr, *d_f16_v3=nullptr;
  uint16_t *h_mask_edge_v1=nullptr, *d_mask_edge_v1=nullptr;
  uint16_t *h_mask_edge_v2=nullptr, *d_mask_edge_v2=nullptr;
  uint16_t *h_mask_edge_v3=nullptr, *d_mask_edge_v3=nullptr;
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
  ACL_CHECK(aclrtMallocHost((void**)&h_mask_edge_v1,SZ_mask_edge));
  ACL_CHECK(aclrtMallocHost((void**)&h_mask_edge_v2,SZ_mask_edge));
  ACL_CHECK(aclrtMallocHost((void**)&h_mask_edge_v3,SZ_mask_edge));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v1,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v2,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v3,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f16_v1,SZ_f16,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f16_v2,SZ_f16,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f16_v3,SZ_f16,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_mask_edge_v1,SZ_mask_edge,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_mask_edge_v2,SZ_mask_edge,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_mask_edge_v3,SZ_mask_edge,ACL_MEM_MALLOC_HUGE_FIRST));
  fsz=SZ_f32; FCK(ReadFile("v1.bin",fsz,h_f32_v1,SZ_f32)&&fsz==SZ_f32,"v1.bin");
  fsz=SZ_f32; FCK(ReadFile("v2.bin",fsz,h_f32_v2,SZ_f32)&&fsz==SZ_f32,"v2.bin");
  fsz=SZ_f32; FCK(ReadFile("v3.bin",fsz,h_f32_v3,SZ_f32)&&fsz==SZ_f32,"v3.bin");
  fsz=SZ_f16; FCK(ReadFile("v1_f16.bin",fsz,h_f16_v1,SZ_f16)&&fsz==SZ_f16,"v1_f16.bin");
  fsz=SZ_f16; FCK(ReadFile("v2_f16.bin",fsz,h_f16_v2,SZ_f16)&&fsz==SZ_f16,"v2_f16.bin");
  fsz=SZ_f16; FCK(ReadFile("v3_f16.bin",fsz,h_f16_v3,SZ_f16)&&fsz==SZ_f16,"v3_f16.bin");
  fsz=SZ_mask_edge; FCK(ReadFile("v1_mask_edge.bin",fsz,h_mask_edge_v1,SZ_mask_edge)&&fsz==SZ_mask_edge,"v1_mask_edge.bin");
  fsz=SZ_mask_edge; FCK(ReadFile("v2_mask_edge.bin",fsz,h_mask_edge_v2,SZ_mask_edge)&&fsz==SZ_mask_edge,"v2_mask_edge.bin");
  fsz=SZ_mask_edge; FCK(ReadFile("v3_mask_edge.bin",fsz,h_mask_edge_v3,SZ_mask_edge)&&fsz==SZ_mask_edge,"v3_mask_edge.bin");
  ACL_CHECK(aclrtMemcpy(d_f32_v1,SZ_f32,h_f32_v1,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v2,SZ_f32,h_f32_v2,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v3,SZ_f32,h_f32_v3,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v1,SZ_f16,h_f16_v1,SZ_f16,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v2,SZ_f16,h_f16_v2,SZ_f16,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f16_v3,SZ_f16,h_f16_v3,SZ_f16,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_mask_edge_v1,SZ_mask_edge,h_mask_edge_v1,SZ_mask_edge,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_mask_edge_v2,SZ_mask_edge,h_mask_edge_v2,SZ_mask_edge,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_mask_edge_v3,SZ_mask_edge,h_mask_edge_v3,SZ_mask_edge,ACL_MEMCPY_HOST_TO_DEVICE));
    LaunchVorDeepMerged(
      d_f32_v1,
      d_f32_v2,
      d_f32_v3,
      d_f16_v1,
      d_f16_v2,
      d_f16_v3,
      d_mask_edge_v1,
      d_mask_edge_v2,
      d_mask_edge_v3,
      stream
  );
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(h_f32_v3,SZ_f32,d_f32_v3,SZ_f32,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_f16_v3,SZ_f16,d_f16_v3,SZ_f16,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_mask_edge_v3,SZ_mask_edge,d_mask_edge_v3,SZ_mask_edge,ACL_MEMCPY_DEVICE_TO_HOST));
  FCK(WriteFile("v3.bin",h_f32_v3,SZ_f32),"v3.bin");
  FCK(WriteFile("v3_f16.bin",h_f16_v3,SZ_f16),"v3_f16.bin");
  FCK(WriteFile("v3_mask_edge.bin",h_mask_edge_v3,SZ_mask_edge),"v3_mask_edge.bin");

cleanup:
  aclrtFree(d_f32_v1);
  aclrtFree(d_f32_v2);
  aclrtFree(d_f32_v3);
  aclrtFree(d_f16_v1);
  aclrtFree(d_f16_v2);
  aclrtFree(d_f16_v3);
  aclrtFree(d_mask_edge_v1);
  aclrtFree(d_mask_edge_v2);
  aclrtFree(d_mask_edge_v3);
  aclrtFreeHost(h_f32_v1);
  aclrtFreeHost(h_f32_v2);
  aclrtFreeHost(h_f32_v3);
  aclrtFreeHost(h_f16_v1);
  aclrtFreeHost(h_f16_v2);
  aclrtFreeHost(h_f16_v3);
  aclrtFreeHost(h_mask_edge_v1);
  aclrtFreeHost(h_mask_edge_v2);
  aclrtFreeHost(h_mask_edge_v3);
  if(stream) aclrtDestroyStream(stream);
  if(deviceSet) aclrtResetDevice(deviceId);
  if(aclInited) aclFinalize();
  return rc;
}
