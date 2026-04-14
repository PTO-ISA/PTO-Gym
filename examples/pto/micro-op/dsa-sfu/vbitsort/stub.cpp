#include <cstdint>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ [aicore] void vbitsort_kernel_f32(__gm__ float *scores,
                                                      __gm__ uint32_t *indices,
                                                      __gm__ uint32_t *output) {
  (void)scores;
  (void)indices;
  (void)output;
}
