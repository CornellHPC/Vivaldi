// C++ standard imports
#include <math.h>

// Library imports
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

// Local imports
#include "linear_kernel.cuh"
#include "utils.cuh"

__global__ void polynonial_kernel_cu_(int64_t mb, DATA_TYPE *B, DATA_TYPE gamma,
                                      DATA_TYPE c, DATA_TYPE r) {
  for (int i = threadIdx.x; i < mb * mb; i += blockDim.x) {
    // we need a for loop in case the block size =/= n^2
    B[i] = -2.0f * powf(gamma * B[i] + c, r);
  }
}

namespace popcorn {

void Kernel::f(DATA_TYPE *B) { std::cerr << "Unimplemented!" << std::endl; }

void PolynomialKernel::f(DATA_TYPE *B) {
  // quick return
  if (mb == 0)
    return;

  // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
  int64_t nthreads = std::min(int64_t(1024), mb * mb);

  polynonial_kernel_cu_<<<1, nthreads>>>(mb, B, gamma, c, r);
}

} // namespace popcorn
