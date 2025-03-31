// C++ standard imports
#include <stdio.h>
#include <cstdio>

// Local imports
#include "argmin_kernel.cuh"

__global__ void argmin_kernel_cu_(int m, int n, int gm, DATA_TYPE* D,
                                  popcorn::Argmin* M) {
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    DATA_TYPE mn = D[j];
    int mni = gm;

    for (int i = 1; i < m; ++i) {
      DATA_TYPE v = D[i * n + j];
      if (v < mn) {
        mn = v;
        mni = i + gm;
      }
    }

    M[j] = popcorn::Argmin{mn, mni};
  }
}

namespace popcorn {

Argmin* ArgminKernel::kernel(int m, int n, int gm, DATA_TYPE* D) {
  // quick return
  if (m == 0 || n == 0)
    return NULL;

  // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
  int nthreads = std::min(int(1024), n);

  // move data to GPU
  DATA_TYPE* dD;
  int d_size = m * n * sizeof(DATA_TYPE);
  cudaMalloc(&dD, d_size);
  cudaMemcpy(dD, D, d_size, cudaMemcpyHostToDevice);

  // create output buffer
  Argmin* dM;
  int m_size = n * sizeof(Argmin);
  cudaMalloc(&dM, m_size);

  argmin_kernel_cu_<<<1, nthreads>>>(m, n, gm, dD, dM);

  Argmin* M = (Argmin*)malloc(m_size);
  cudaMemcpy(M, dM, m_size, cudaMemcpyDeviceToHost);

  cudaFree(dD);
  cudaFree(dM);

  return M;
}

}  // namespace popcorn
