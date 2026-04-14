// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/pset-pattern
// family: materialization-predicate
// target_ops: pto.pset_b16, pto.pset_b32, pto.pset_b8
// scenarios: pattern-mask, pat-all, pat-vl
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
extern "C" __global__ [aicore] void pset_pattern_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
