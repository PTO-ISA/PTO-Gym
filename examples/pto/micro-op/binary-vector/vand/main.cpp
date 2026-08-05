// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Merged vand test case.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
using namespace PtoTestCommon;

#define ACL_CHECK(expr) do {   const aclError _r=(expr);   if(_r!=ACL_SUCCESS){std::fprintf(stderr,"[ERROR] %s:%d acle=%d\n",#expr,__LINE__,(int)_r);rc=1;goto cleanup;} }while(0)
#define FCK(expr,path) do{if(!(expr)){std::fprintf(stderr,"[ERROR] file:%s\n",path);rc=1;goto cleanup;}}while(0)




void LaunchVandDeepMerged(uint16_t * p0, uint16_t * p1, uint16_t * p2, uint16_t * p3, uint16_t * p4, uint16_t * p5, uint8_t * p6, uint8_t * p7, uint8_t * p8, uint8_t * p9, uint8_t * p10, uint8_t * p11, uint8_t * p12, uint8_t * p13, uint8_t * p14, uint8_t * p15, void *stream);
int main() {
  constexpr size_t SZ_f32 = 2048;
  constexpr size_t SZ_mask_edge = 2048;
  constexpr size_t SZ_lowp = 256;

  uint16_t *h_f32_v1=nullptr, *d_f32_v1=nullptr;
  uint16_t *h_f32_v2=nullptr, *d_f32_v2=nullptr;
  uint16_t *h_f32_v3=nullptr, *d_f32_v3=nullptr;
  uint16_t *h_mask_edge_v1=nullptr, *d_mask_edge_v1=nullptr;
  uint16_t *h_mask_edge_v2=nullptr, *d_mask_edge_v2=nullptr;
  uint16_t *h_mask_edge_v3=nullptr, *d_mask_edge_v3=nullptr;
  uint8_t *h_f8_v1=nullptr, *d_f8_v1=nullptr;
  uint8_t *h_f8_v2=nullptr, *d_f8_v2=nullptr;
  uint8_t *h_f8_and=nullptr, *d_f8_and=nullptr;
  uint8_t *h_f8_xor=nullptr, *d_f8_xor=nullptr;
  uint8_t *h_f8_or=nullptr, *d_f8_or=nullptr;
  uint8_t *h_hif8_v1=nullptr, *d_hif8_v1=nullptr;
  uint8_t *h_hif8_v2=nullptr, *d_hif8_v2=nullptr;
  uint8_t *h_hif8_and=nullptr, *d_hif8_and=nullptr;
  uint8_t *h_hif8_xor=nullptr, *d_hif8_xor=nullptr;
  uint8_t *h_hif8_or=nullptr, *d_hif8_or=nullptr;
  int rc=0; bool aclInited=false,deviceSet=false; int deviceId=0; aclrtStream stream=nullptr; size_t fsz=0;
  ACL_CHECK(aclInit(nullptr)); aclInited=true;
  if(const char*e=std::getenv("ACL_DEVICE_ID")) deviceId=std::atoi(e);
  ACL_CHECK(aclrtSetDevice(deviceId)); deviceSet=true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v1,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v2,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_f32_v3,SZ_f32));
  ACL_CHECK(aclrtMallocHost((void**)&h_mask_edge_v1,SZ_mask_edge));
  ACL_CHECK(aclrtMallocHost((void**)&h_mask_edge_v2,SZ_mask_edge));
  ACL_CHECK(aclrtMallocHost((void**)&h_mask_edge_v3,SZ_mask_edge));
  ACL_CHECK(aclrtMallocHost((void**)&h_f8_v1,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_f8_v2,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_f8_and,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_f8_xor,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_f8_or,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_hif8_v1,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_hif8_v2,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_hif8_and,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_hif8_xor,SZ_lowp));
  ACL_CHECK(aclrtMallocHost((void**)&h_hif8_or,SZ_lowp));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v1,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v2,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f32_v3,SZ_f32,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_mask_edge_v1,SZ_mask_edge,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_mask_edge_v2,SZ_mask_edge,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_mask_edge_v3,SZ_mask_edge,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f8_v1,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f8_v2,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f8_and,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f8_xor,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_f8_or,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_hif8_v1,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_hif8_v2,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_hif8_and,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_hif8_xor,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void**)&d_hif8_or,SZ_lowp,ACL_MEM_MALLOC_HUGE_FIRST));
  fsz=SZ_f32; FCK(ReadFile("v1.bin",fsz,h_f32_v1,SZ_f32)&&fsz==SZ_f32,"v1.bin");
  fsz=SZ_f32; FCK(ReadFile("v2.bin",fsz,h_f32_v2,SZ_f32)&&fsz==SZ_f32,"v2.bin");
  fsz=SZ_f32; FCK(ReadFile("v3.bin",fsz,h_f32_v3,SZ_f32)&&fsz==SZ_f32,"v3.bin");
  fsz=SZ_mask_edge; FCK(ReadFile("v1_mask_edge.bin",fsz,h_mask_edge_v1,SZ_mask_edge)&&fsz==SZ_mask_edge,"v1_mask_edge.bin");
  fsz=SZ_mask_edge; FCK(ReadFile("v2_mask_edge.bin",fsz,h_mask_edge_v2,SZ_mask_edge)&&fsz==SZ_mask_edge,"v2_mask_edge.bin");
  fsz=SZ_mask_edge; FCK(ReadFile("v3_mask_edge.bin",fsz,h_mask_edge_v3,SZ_mask_edge)&&fsz==SZ_mask_edge,"v3_mask_edge.bin");
  fsz=SZ_lowp; FCK(ReadFile("v1_f8.bin",fsz,h_f8_v1,SZ_lowp)&&fsz==SZ_lowp,"v1_f8.bin");
  fsz=SZ_lowp; FCK(ReadFile("v2_f8.bin",fsz,h_f8_v2,SZ_lowp)&&fsz==SZ_lowp,"v2_f8.bin");
  fsz=SZ_lowp; FCK(ReadFile("v3_f8_and.bin",fsz,h_f8_and,SZ_lowp)&&fsz==SZ_lowp,"v3_f8_and.bin");
  fsz=SZ_lowp; FCK(ReadFile("v4_f8_xor.bin",fsz,h_f8_xor,SZ_lowp)&&fsz==SZ_lowp,"v4_f8_xor.bin");
  fsz=SZ_lowp; FCK(ReadFile("v5_f8_or.bin",fsz,h_f8_or,SZ_lowp)&&fsz==SZ_lowp,"v5_f8_or.bin");
  fsz=SZ_lowp; FCK(ReadFile("v1_hif8.bin",fsz,h_hif8_v1,SZ_lowp)&&fsz==SZ_lowp,"v1_hif8.bin");
  fsz=SZ_lowp; FCK(ReadFile("v2_hif8.bin",fsz,h_hif8_v2,SZ_lowp)&&fsz==SZ_lowp,"v2_hif8.bin");
  fsz=SZ_lowp; FCK(ReadFile("v3_hif8_and.bin",fsz,h_hif8_and,SZ_lowp)&&fsz==SZ_lowp,"v3_hif8_and.bin");
  fsz=SZ_lowp; FCK(ReadFile("v4_hif8_xor.bin",fsz,h_hif8_xor,SZ_lowp)&&fsz==SZ_lowp,"v4_hif8_xor.bin");
  fsz=SZ_lowp; FCK(ReadFile("v5_hif8_or.bin",fsz,h_hif8_or,SZ_lowp)&&fsz==SZ_lowp,"v5_hif8_or.bin");
  ACL_CHECK(aclrtMemcpy(d_f32_v1,SZ_f32,h_f32_v1,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v2,SZ_f32,h_f32_v2,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f32_v3,SZ_f32,h_f32_v3,SZ_f32,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_mask_edge_v1,SZ_mask_edge,h_mask_edge_v1,SZ_mask_edge,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_mask_edge_v2,SZ_mask_edge,h_mask_edge_v2,SZ_mask_edge,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_mask_edge_v3,SZ_mask_edge,h_mask_edge_v3,SZ_mask_edge,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f8_v1,SZ_lowp,h_f8_v1,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f8_v2,SZ_lowp,h_f8_v2,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f8_and,SZ_lowp,h_f8_and,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f8_xor,SZ_lowp,h_f8_xor,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_f8_or,SZ_lowp,h_f8_or,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_hif8_v1,SZ_lowp,h_hif8_v1,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_hif8_v2,SZ_lowp,h_hif8_v2,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_hif8_and,SZ_lowp,h_hif8_and,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_hif8_xor,SZ_lowp,h_hif8_xor,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(d_hif8_or,SZ_lowp,h_hif8_or,SZ_lowp,ACL_MEMCPY_HOST_TO_DEVICE));
    LaunchVandDeepMerged(
      d_f32_v1,
      d_f32_v2,
      d_f32_v3,
      d_mask_edge_v1,
      d_mask_edge_v2,
      d_mask_edge_v3,
      d_f8_v1,
      d_f8_v2,
      d_f8_and,
      d_f8_xor,
      d_f8_or,
      d_hif8_v1,
      d_hif8_v2,
      d_hif8_and,
      d_hif8_xor,
      d_hif8_or,
      stream
  );
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(h_f32_v3,SZ_f32,d_f32_v3,SZ_f32,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_mask_edge_v3,SZ_mask_edge,d_mask_edge_v3,SZ_mask_edge,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_f8_and,SZ_lowp,d_f8_and,SZ_lowp,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_f8_xor,SZ_lowp,d_f8_xor,SZ_lowp,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_f8_or,SZ_lowp,d_f8_or,SZ_lowp,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_hif8_and,SZ_lowp,d_hif8_and,SZ_lowp,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_hif8_xor,SZ_lowp,d_hif8_xor,SZ_lowp,ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(h_hif8_or,SZ_lowp,d_hif8_or,SZ_lowp,ACL_MEMCPY_DEVICE_TO_HOST));
  FCK(WriteFile("v3.bin",h_f32_v3,SZ_f32),"v3.bin");
  FCK(WriteFile("v3_mask_edge.bin",h_mask_edge_v3,SZ_mask_edge),"v3_mask_edge.bin");
  FCK(WriteFile("v3_f8_and.bin",h_f8_and,SZ_lowp),"v3_f8_and.bin");
  FCK(WriteFile("v4_f8_xor.bin",h_f8_xor,SZ_lowp),"v4_f8_xor.bin");
  FCK(WriteFile("v5_f8_or.bin",h_f8_or,SZ_lowp),"v5_f8_or.bin");
  FCK(WriteFile("v3_hif8_and.bin",h_hif8_and,SZ_lowp),"v3_hif8_and.bin");
  FCK(WriteFile("v4_hif8_xor.bin",h_hif8_xor,SZ_lowp),"v4_hif8_xor.bin");
  FCK(WriteFile("v5_hif8_or.bin",h_hif8_or,SZ_lowp),"v5_hif8_or.bin");

cleanup:
  aclrtFree(d_f32_v1);
  aclrtFree(d_f32_v2);
  aclrtFree(d_f32_v3);
  aclrtFree(d_mask_edge_v1);
  aclrtFree(d_mask_edge_v2);
  aclrtFree(d_mask_edge_v3);
  aclrtFree(d_f8_v1);
  aclrtFree(d_f8_v2);
  aclrtFree(d_f8_and);
  aclrtFree(d_f8_xor);
  aclrtFree(d_f8_or);
  aclrtFree(d_hif8_v1);
  aclrtFree(d_hif8_v2);
  aclrtFree(d_hif8_and);
  aclrtFree(d_hif8_xor);
  aclrtFree(d_hif8_or);
  aclrtFreeHost(h_f32_v1);
  aclrtFreeHost(h_f32_v2);
  aclrtFreeHost(h_f32_v3);
  aclrtFreeHost(h_mask_edge_v1);
  aclrtFreeHost(h_mask_edge_v2);
  aclrtFreeHost(h_mask_edge_v3);
  aclrtFreeHost(h_f8_v1);
  aclrtFreeHost(h_f8_v2);
  aclrtFreeHost(h_f8_and);
  aclrtFreeHost(h_f8_xor);
  aclrtFreeHost(h_f8_or);
  aclrtFreeHost(h_hif8_v1);
  aclrtFreeHost(h_hif8_v2);
  aclrtFreeHost(h_hif8_and);
  aclrtFreeHost(h_hif8_xor);
  aclrtFreeHost(h_hif8_or);
  if(stream) aclrtDestroyStream(stream);
  if(deviceSet) aclrtResetDevice(deviceId);
  if(aclInited) aclFinalize();
  return rc;
}
