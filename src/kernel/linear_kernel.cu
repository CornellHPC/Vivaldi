#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "linear_kernel.cuh"
#include "utils.cuh"

__global__ void polynonial_kernel_cu_(int64_t n, DATA_TYPE* B, DATA_TYPE gamma,
                                      DATA_TYPE c, DATA_TYPE r) {
  for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
    // we need a for loop in case the block size =/= n^2
    B[i] = -2.0f * powf(gamma * B[i] + c, r);
  }
}

void kernel::polynomial_kernel(int64_t n, DATA_TYPE* B, DATA_TYPE gamma,
                               DATA_TYPE c, DATA_TYPE r) {
  // quick return
  if (n == 0) return;

  // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
  int64_t nthreads = std::min(int64_t(1024), n * n);

  polynonial_kernel_cu_<<<1, nthreads>>>(n, B, gamma, c, r);
}