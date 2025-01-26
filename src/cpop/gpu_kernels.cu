#include "gpu_kernels.cuh"

#include "stdio.h"

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void polynomial_kernel(int64_t m, int64_t n, float* B, float gamma,
                                  float c, float r) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n;
       i += blockDim.x * gridDim.x) {
    // this for loop runs once unless the matrix is extraordinarily large
    B[i] = powf(gamma * B[i] + c, r);
  }
}

namespace cpop {

void launch_polynomial_kernel(int64_t m, int64_t n, float* B, float gamma,
                              float c, float r) {
  if (m == 0 || n == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (m * n + nthreads - 1) / nthreads);

  polynomial_kernel<<<nblocks, nthreads>>>(m, n, B, gamma, c, r);
}

__global__ void z_vector_kernel(int64_t t, float* z, int64_t* assignments,
                                float* ET) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t;
       i += blockDim.x * gridDim.x) {
    // this for loop runs once unless the matrix is extraordinarily large
    // i represents the different points (e.g. columns of ET)
    // we fetch the element at (assignments[i], i) from ET which is basically as
    // mask of ET using V + flattening column-wise
    z[i] = ET[t * assignments[i] + i];
  }
}

void launch_z_kernel(int64_t t, float* z, int64_t* assignments, float* ET) {
  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (t + nthreads - 1) / nthreads);

  z_vector_kernel<<<nblocks, nthreads>>>(t, z, assignments, ET);
}

}  // namespace cpop
