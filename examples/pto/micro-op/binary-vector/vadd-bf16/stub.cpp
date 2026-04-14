// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vadd-bf16
// family: binary-vector
// target_ops: pto.vadd
// scenarios: core-bf16, full-mask
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vadd_bf16_kernel(__gm__ bfloat16_t *v1,
                                                   __gm__ bfloat16_t *v2,
                                                   __gm__ bfloat16_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
