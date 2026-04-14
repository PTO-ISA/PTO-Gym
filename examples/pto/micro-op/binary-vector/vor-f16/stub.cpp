// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vor-f16
// family: binary-vector
// target_ops: pto.vor
// scenarios: core-f16, full-mask
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------
#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vor_f16_kernel(__gm__ half *v1,
                                                 __gm__ half *v2,
                                                 __gm__ half *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
