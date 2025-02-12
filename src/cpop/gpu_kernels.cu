#include <float.h>
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>

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
                              int* global_cluster_sizes, int64_t m) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
       i += blockDim.x * gridDim.x) {
    // this for loop runs once unless the matrix is extraordinarily large

    // todo: the global_assignments[i] call is coalesced, but the global_cluster_sizes[...] call is not!
    V_global_values[i] = 1.0f / global_cluster_sizes[global_assignments[i]];
  }
}

// TODO: this doesnt work cause I misunderstood how shared memory works
// __global__ void d_kernel(int64_t k, int64_t t, float* dE, float* dc,
//                          int64_t* a) {
//   int64_t i = blockIdx.x * blockDim.x + threadIdx.x;  // rows of E
//   if (i < k) {
//     __shared__ float shared_min[blockDim.y];
//     __shared__ long long int shared_min_idx[blockDim.y];

//     for (int64_t j = blockIdx.y * blockDim.y + threadIdx.y; j < t;
//          j += blockDim.y * gridDim.y) {  // cols of E
//       float value = dc[i] - 2 * dE[i * t + j];
//       dE[i * t + j] = value;

//       if (i == 0) {
//         shared_min[j] = FLT_MAX;
//         shared_min_idx[j] = INT_MAX;
//       }

//       __syncthreads();
//       atomicMinFloat(&shared_min[j], value);
//       __syncthreads();
//       if (shared_min[j] == value) {
//         // Taking the minimum here for stability. If two values of E in the same column are
//         // exactly the same (unlikely in floating point), then take the smallest row by index
//         // value to be the argmin.
//         atomicMin(&shared_min_idx[j], i);
//       }
//       __syncthreads();

//       if (i == 0) {
//         a[j] = shared_min_idx[j];
//       }
//     }
//   }
// }

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
                          int* global_cluster_sizes, int64_t m) {
  if (m == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (m + nthreads - 1) / nthreads);
  reinit_kernel<<<nblocks, nthreads>>>(V_global_values, global_assignments,
                                       global_cluster_sizes, m);
}

// void launch_d_kernel(int64_t k, int64_t t, float* dE, float* dc, int64_t* a) {
//   // 1024 max threads for current CUDA compute capability (<= 7.5)
//   // 16x16 blocks, with upwards round for more coverage
//   // block cap is set to prevent overflow
//   int clusters_tsize = 16;
//   int points_tsize = 16;
//   dim3 nthreads(clusters_tsize, points_tsize);

//   // t may be very large so it may span across multiple grids
//   // k is assumed to be smaller than INT_MAX. If it's not, this would greatly
//   // complicate the argmin kernel, so we assume that this is not the case
//   assert(k < INT_MAX && "k is too large");
//   // long long int test_a = 7;  // TODO: rm
//   // int64_t test_b = 7;
//   // assert(sizeof(test_a) == sizeof(test_b) &&
//   //        "The system does not equate long long int with int64_t");
//   dim3 nblocks(
//       (k + clusters_tsize - 1) / clusters_tsize,
//       std::min(int64_t(1048576), (t + points_tsize - 1) / points_tsize));

//   // d_kernel<<<nblocks, nthreads>>>(k, t, dE, dc, a);
// }

void scan(int64_t* d_in, int64_t* d_out, int64_t k) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                k + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                k + 1);
}

}  // namespace cpop
