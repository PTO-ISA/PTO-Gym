// -----------------------------------------------------------------------------
// case: micro-op/vec-scalar/vadds-i16-signed-overflow
// family: vec-scalar
// target_ops: pto.vadds
// scenarios: core-i16-signed, full-mask, scalar-operand, integer-overflow
// -----------------------------------------------------------------------------
#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void
vadds_i16_signed_overflow_kernel(__gm__ int16_t *v1, __gm__ int16_t *v2) {
  (void)v1;
  (void)v2;
}
