// -----------------------------------------------------------------------------
// case: micro-op/gather-scatter/vgather2_bc
// family: gather-scatter
// target_ops: pto.vgather2_bc
// scenarios: core-f32, full-mask, non-contiguous, masked-gather, load-effect-validation, no-alias
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

// The runtime launcher resolves the real device implementation from the
// embedded aibinary. The host-side fatobj still needs a concrete kernel symbol
// with the final ABI name, but it does not need the original EmitC body.
extern "C" __global__ AICORE void vgather2_bc_kernel_2d(__gm__ float *v1,
                                                        __gm__ int *v2,
                                                        __gm__ float *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
