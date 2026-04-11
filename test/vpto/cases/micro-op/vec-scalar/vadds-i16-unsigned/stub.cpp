// -----------------------------------------------------------------------------
// case: micro-op/vec-scalar/vadds-i16-unsigned
// family: vec-scalar
// target_ops: pto.vadds
// scenarios: core-i16-unsigned, full-mask, scalar-operand
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>
#include <stdint.h>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vadds_i16_unsigned_kernel(__gm__ uint16_t *v1,
                                                            __gm__ uint16_t *v2) {
  (void)v1;
  (void)v2;
}
