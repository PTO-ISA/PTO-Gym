// -----------------------------------------------------------------------------
// case: micro-op/vector-load-store/vldsx2-layout-check
// family: vector-load-store
// target_ops: pto.vldsx2
// scenarios: core-f32, full-mask, paired-roundtrip, dintlv, lane-order
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
extern "C" __global__ AICORE void vldx2_layout_check_kernel(__gm__ float *v1,
                                                            __gm__ float *v2) {
  (void)v1;
  (void)v2;
}
