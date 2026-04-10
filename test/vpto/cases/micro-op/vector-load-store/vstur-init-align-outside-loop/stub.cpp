// -----------------------------------------------------------------------------
// case: micro-op/vector-load-store/vstur-init-align-outside-loop
// family: vector-load-store
// target_ops: pto.vstur
// scenarios: core-f32, full-mask, unaligned, state-update, init-align-outside-loop
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

extern "C" __global__ AICORE void
vstur_init_align_outside_loop_kernel_2d(__gm__ float *v1, __gm__ float *v2) {
  (void)v1;
  (void)v2;
}
