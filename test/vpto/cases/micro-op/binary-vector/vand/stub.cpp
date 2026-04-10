// -----------------------------------------------------------------------------
// case: micro-op/binary-vector/vand
// family: binary-vector
// target_ops: pto.vand
// scenarios: core-i16-unsigned, full-mask
// -----------------------------------------------------------------------------
#include <cstdint>
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void vand_i16_unsigned_kernel(__gm__ uint16_t *v1,
                                                           __gm__ uint16_t *v2,
                                                           __gm__ uint16_t *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
