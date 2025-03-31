// C++ standard imports
#include <math.h>

// Library imports
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

// Local imports
#include "dist_kernel.cuh"

__global__ void dist_kernel_cu_(int64_t m, int64_t n, DATA_TYPE* ET,
                                DATA_TYPE* c) {
  for (int64_t i = threadIdx.x; i < m * n; i += blockDim.x) {
    // we need a for loop in case the block size =/= n^2
    ET[i] = c[i / n] - 2 * ET[i];
  }
}

namespace popcorn {

DATA_TYPE* DistKernel::kernel(int64_t m, int64_t n, DATA_TYPE* ET,
                              DATA_TYPE* c) {
  // quick return
  if (m == 0 || n == 0)
    return NULL;

  // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
  int64_t nthreads = std::min(int64_t(1024), m * n);

  // move data to GPU
  DATA_TYPE* dET;
  int64_t d_size = m * n * sizeof(DATA_TYPE);
  cudaMalloc(&dET, d_size);
  cudaMemcpy(dET, ET, d_size, cudaMemcpyHostToDevice);
  DATA_TYPE* dc;
  int64_t c_size = m * sizeof(DATA_TYPE);
  cudaMalloc(&dc, c_size);
  cudaMemcpy(dc, c, c_size, cudaMemcpyHostToDevice);

  dist_kernel_cu_<<<1, nthreads>>>(m, n, dET, dc);

  DATA_TYPE* D = (DATA_TYPE*)malloc(d_size);
  cudaMemcpy(D, dET, d_size, cudaMemcpyDeviceToHost);

  cudaFree(dET);
  cudaFree(dc);

  return D;
}

}  // namespace popcorn
