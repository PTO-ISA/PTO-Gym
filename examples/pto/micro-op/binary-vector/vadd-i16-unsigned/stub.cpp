// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vadd-i16-unsigned
// family: binary-vector
// target_ops: pto.vadd
// scenarios: core-i16-unsigned, full-mask
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vadd_i16_unsigned_kernel(__gm__ uint16_t *v1,
                                                           __gm__ uint16_t *v2,
                                                           __gm__ uint16_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
