// -----------------------------------------------------------------------------
// case: micro-op/vector-load-store/vsstb
// family: vector-load-store
// target_ops: pto.vsstb
// scenarios: core-f32, full-mask, block-strided-store, block-mask
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
extern "C" __global__ AICORE void vsstb_kernel_2d(__gm__ float *v1,
                                                  __gm__ float *v2) {
  (void)v1;
  (void)v2;
}
