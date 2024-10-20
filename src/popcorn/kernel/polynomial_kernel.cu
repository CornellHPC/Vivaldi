// C++ standard imports
#include <math.h>

// Library imports
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

// Local imports
#include "polynomial_kernel.cuh"
#include "utils.cuh"

__global__ void polynonial_kernel_cu_(int64_t m, int64_t n, DATA_TYPE *B,
                                      DATA_TYPE gamma, DATA_TYPE c,
                                      DATA_TYPE r) {
  for (int i = threadIdx.x; i < m * n; i += blockDim.x) {
    // we need a for loop in case the block size =/= n^2
    B[i] = powf(gamma * B[i] + c, r);
  }
}

namespace popcorn {

void PolynomialKernel::f(int64_t m, int64_t n, DATA_TYPE *B) {
  // quick return
  if (m == 0 || n == 0)
    return;

  // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
  int64_t nthreads = std::min(int64_t(1024), m * n);

  polynonial_kernel_cu_<<<1, nthreads>>>(m, n, B, gamma, c, r);
}

} // namespace popcorn
