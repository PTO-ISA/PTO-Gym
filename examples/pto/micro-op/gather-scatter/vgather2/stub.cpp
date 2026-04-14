// -----------------------------------------------------------------------------
// case: micro-op/gather-scatter/vgather2
// family: gather-scatter
// target_ops: pto.vgather2
// scenarios: core-f32, full-mask, non-contiguous, explicit-index-pattern, load-effect-validation, no-alias
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

// The runtime launcher resolves the real device implementation from the
// embedded aibinary. The host-side fatobj still needs a concrete kernel symbol
// with the final ABI name, but it does not need the original EmitC body.
extern "C" __global__ [aicore] void vgather2_kernel_2d(__gm__ float *v1,
                                                     __gm__ int *v2,
                                                     __gm__ float *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
