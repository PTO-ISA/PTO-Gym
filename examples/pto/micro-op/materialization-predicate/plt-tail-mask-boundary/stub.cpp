// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/plt-tail-mask-boundary
// family: materialization-predicate
// target_ops: pto.plt_b16, pto.plt_b32, pto.plt_b8
// scenarios: tail-mask, scalar-carry-out, boundary

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void plt_tail_mask_boundary_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
