// C++ standard imports
#include <stdio.h>
#include <cstdio>

// Local imports
#include "argmin_kernel.cuh"

__global__ void argmin_kernel_cu_(int64_t m, int64_t n, DATA_TYPE* D,
                                  popcorn::Argmin* M) {
  for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
    DATA_TYPE mn = D[j];
    int64_t mni = 0;

    for (int64_t i = 1; i < m; ++i) {
      DATA_TYPE v = D[i * n + j];
      if (v < mn) {
        mn = v;
        mni = i;
      }
    }

    M[j] = popcorn::Argmin{mn, mni};
  }
}

namespace popcorn {

Argmin* ArgminKernel::kernel(int64_t m, int64_t n, DATA_TYPE* D) {
  // quick return
  if (m == 0 || n == 0)
    return NULL;

  // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
  int64_t nthreads = std::min(int64_t(1024), n);

  // move data to GPU
  DATA_TYPE* dD;
  int64_t d_size = m * n * sizeof(DATA_TYPE);
  cudaMalloc(&dD, d_size);
  cudaMemcpy(dD, D, d_size, cudaMemcpyHostToDevice);

  // create output buffer
  Argmin* dM;
  int64_t m_size = n * sizeof(Argmin);
  cudaMalloc(&dM, m_size);

  argmin_kernel_cu_<<<1, nthreads>>>(m, n, dD, dM);

  Argmin* M = (Argmin*)malloc(m_size);
  cudaMemcpy(M, dM, m_size, cudaMemcpyDeviceToHost);

  cudaFree(dD);
  cudaFree(dM);

  return M;
}

}  // namespace popcorn
