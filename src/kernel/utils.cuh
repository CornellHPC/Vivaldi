#ifndef DISTRIBUTED_POPCORN_KERNEL_UTILS_H
#define DISTRIBUTED_POPCORN_KERNEL_UTILS_H

#include <stdio.h>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char* file, int line,
               bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#endif  // DISTRIBUTED_POPCORN_KERNEL_UTILS_H