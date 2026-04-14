// -----------------------------------------------------------------------------
// case: micro-op/materialization-predicate/ppack-punpack
// family: materialization-predicate
// target_ops: pto.ppack, pto.punpack
// scenarios: pack-unpack-roundtrip

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void ppack_punpack_kernel_2d(__gm__ uint32_t *v1) {
  (void)v1;
}
