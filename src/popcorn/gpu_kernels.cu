#include "gpu_kernels.cuh"

__global__ void polynomial_kernel(int64_t m, int64_t n, float* B, float gamma,
                                  float c, float r) {
  // Grid-stride loop
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n;
       i += blockDim.x * gridDim.x) {
    B[i] = powf(gamma * B[i] + c, r);
  }
}

void popcorn::launch_polynomial_kernel(int64_t m, int64_t n, float* B,
                                       float gamma, float c, float r) {
  if (m == 0 || n == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  int64_t nthreads = std::min(int64_t(1024), m * n);
  int64_t nblocks = (m * n + nthreads - 1) / nthreads;

  polynomial_kernel<<<nblocks, nthreads>>>(m, n, B, gamma, c, r);
}