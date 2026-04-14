// -----------------------------------------------------------------------------
// case: micro-op/conversion/vcvt-f16-to-f32-part-even
// family: conversion
// target_ops: pto.vcvt
// scenarios: f16-to-f32, full-mask, part-even
// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is
// still a valid test conclusion in the current coverage-first phase.
// -----------------------------------------------------------------------------

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

// The runtime launcher resolves the real device implementation from the
// embedded aibinary. The host-side fatobj still needs a concrete kernel symbol
// with the final ABI name, but it does not need the original EmitC body.
extern "C" __global__ [aicore] void vcvt_f16_to_f32_part_even_kernel_2d(__gm__ half *v1,
                                                                      __gm__ float *v2) {
  (void)v1;
  (void)v2;
}
