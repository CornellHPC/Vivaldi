#include <float.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <cassert>

#include "gpu_kernels.cuh"

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

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
  float old;
  old = !signbit(value)
            ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMax((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

__global__ void polynomial_kernel(int64_t m, int64_t n, float* B, float gamma,
                                  float c, float r) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n;
       i += blockDim.x * gridDim.x) {
    // this for loop runs once unless the matrix is extraordinarily large
    B[i] = powf(gamma * B[i] + c, r);
  }
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

__global__ void argmin_kernel(int64_t k, int64_t t, float* dE, float* dc,
                              int64_t* local_assignments,
                              int* local_cluster_sizes) {
  for (int64_t point = blockIdx.x * blockDim.x + threadIdx.x; point < k;
       point += blockDim.x * gridDim.x) {
    local_cluster_sizes[point] = 0;
  }

  for (int64_t point = blockIdx.x * blockDim.x + threadIdx.x; point < t;
       point += blockDim.x * gridDim.x) {
    // printf("Argmin kernel started with point %lld\n", point);
    float min = FLT_MAX;
    int64_t min_cluster = 0;
    for (int64_t cluster = 0; cluster < k; ++cluster) {
      float value = dc[cluster] - 2 * dE[cluster * t + point];
      // printf("C %f E %f (idx %lld) -> Value %f\n", dc[cluster],
      //        dE[cluster * t + point], cluster * t + point, value);
      if (value < min) {
        min = value;
        min_cluster = cluster;
      }
    }
    // printf("Min %f idx %lld\n", min, min_cluster);
    local_assignments[point] = min_cluster;
    atomicAdd(&local_cluster_sizes[min_cluster], 1);
    // printf("Finished argmin kernel\n");
  }
}

__global__ void reinit_kernel(float* V_global_values,
                              int64_t* global_assignments,
                              int* global_cluster_sizes, int64_t k, int64_t m,
                              bool sparse) {
  if (sparse) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += blockDim.x * gridDim.x) {
      // this for loop runs once unless the matrix is extraordinarily large

      // todo: the global_assignments[i] call is coalesced, but the global_cluster_sizes[...] call is not!
      V_global_values[i] = 1.0f / global_cluster_sizes[global_assignments[i]];
    }
  } else {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < k * m;
         i += blockDim.x * gridDim.x) {
      int64_t point = i % m;
      int64_t cluster = i / m;

      V_global_values[i] = 0.0f;
      if (global_assignments[point] == cluster) {
        V_global_values[i] = 1.0f / global_cluster_sizes[cluster];
      }
    }
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

void launch_z_kernel(int64_t t, float* z, int64_t* assignments, float* ET) {
  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (t + nthreads - 1) / nthreads);

  z_vector_kernel<<<nblocks, nthreads>>>(t, z, assignments, ET);
}

void launch_argmin_kernel(int64_t k, int64_t t, float* dE, float* dc,
                          int64_t* local_assignments,
                          int* local_cluster_sizes) {
  if (k == 0 || t == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (t + nthreads - 1) / nthreads);
  argmin_kernel<<<nblocks, nthreads>>>(k, t, dE, dc, local_assignments,
                                       local_cluster_sizes);
}

void launch_reinit_kernel(float* V_global_values, int64_t* global_assignments,
                          int* global_cluster_sizes, int64_t k, int64_t m,
                          bool sparse) {
  if (k == 0 || m == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (m + nthreads - 1) / nthreads);
  reinit_kernel<<<nblocks, nthreads>>>(V_global_values, global_assignments,
                                       global_cluster_sizes, k, m, sparse);
}

bool test_convergence_equality(int64_t* assignments, int64_t* prev_assignments,
                               int64_t t) {
  std::cout << "testing convergence between two vectors" << std::endl;
  int64_t* h_assignments = new int64_t[t];
  gpuErrchk(cudaMemcpy(h_assignments, assignments, t * sizeof(int64_t), cudaMemcpyDeviceToHost));
  for (int64_t i = 0; i < t; ++i) {
    std::cout << h_assignments[i] << " ";
  }
  std::cout << std::endl;
  delete[] h_assignments;
  int64_t* h_prev_assignments = new int64_t[t];
  gpuErrchk(cudaMemcpy(h_prev_assignments, prev_assignments, t * sizeof(int64_t), cudaMemcpyDeviceToHost));
  for (int64_t i = 0; i < t; ++i) {
    std::cout << h_prev_assignments[i] << " ";
  }
  std::cout << std::endl;
  delete[] h_prev_assignments;

  thrust::device_ptr<int64_t> t_assignments =
      thrust::device_pointer_cast(assignments);
  thrust::device_ptr<int64_t> t_prev_assignments =
      thrust::device_pointer_cast(prev_assignments);
  return thrust::equal(t_assignments, t_assignments + t, t_prev_assignments);
}

}  // namespace cpop
