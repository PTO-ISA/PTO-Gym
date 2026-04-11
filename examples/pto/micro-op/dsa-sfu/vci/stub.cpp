// -----------------------------------------------------------------------------
// case: micro-op/dsa-sfu/vci
// family: dsa-sfu / conversion
// target_ops: pto.vci
// scenarios: index-generation
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

extern "C" __global__ AICORE void vci_kernel_2d(__gm__ int *v1,
                                                __gm__ int *v2) {
  (void)v1;
  (void)v2;
}
