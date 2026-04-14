// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vshr-i16-signed
// family: binary-vector
// target_ops: pto.vshr
// scenarios: core-i16-signed, full-mask
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

extern "C" __global__ [aicore] void vshr_i16_signed_kernel(__gm__ int16_t *v1,
                                                         __gm__ int16_t *v2,
                                                         __gm__ int16_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
