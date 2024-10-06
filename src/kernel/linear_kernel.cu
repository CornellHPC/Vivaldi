#include <cstdint>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "linear_kernel.cuh"

template <typename T>
__global__ void polynonial_kernel_cu_(int64_t n, T *B, T gamma, T c, T r) {
  for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
    // we need a for loop in case the block size =/= n^2
    B[i] = ((T)(-2)) * powf(gamma * B[i] + c, r);
  }
}

template <typename T>
void kernel::polynomial_kernel(int64_t n, T *B, T gamma, T c, T r) {
  // quick return
  if (n == 0)
    return;

  // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
  int64_t nthreads = std::min(int64_t(1024), n * n);

  polynonial_kernel_cu_<<<1, nthreads>>>(n, B, gamma, c, r);
}

// Explicit template instantiations
template void kernel::polynomial_kernel(int64_t, float *, float, float, float);
template void kernel::polynomial_kernel(int64_t, double *, double, double,
                                        double);
template void kernel::polynomial_kernel(int64_t, int *, int, int, int);
template void kernel::polynomial_kernel(int64_t, int64_t *, int64_t, int64_t,
                                        int64_t);
