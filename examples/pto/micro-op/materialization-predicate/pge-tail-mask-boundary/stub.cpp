// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pge-tail-mask-boundary
// family: materialization-predicate
// target_ops: pto.pge_b16, pto.pge_b32, pto.pge_b8
// scenarios: tail-mask, boundary

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void pge_tail_mask_boundary_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
