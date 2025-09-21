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

__global__ void z_vector_kernel(int64_t t, float* z, int* assignments,
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
                              int* local_assignments, int* local_cluster_sizes,
                              bool* converged,
                              float* local_k_means_objective_score,
                              float* local_k_means_objective_delta,
                              float* prev_point_to_cluster_distances) {
  for (int64_t point = blockIdx.x * blockDim.x + threadIdx.x; point < k;
       point += blockDim.x * gridDim.x) {
    local_cluster_sizes[point] = 0;
  }

  for (int64_t point = blockIdx.x * blockDim.x + threadIdx.x; point < t;
       point += blockDim.x * gridDim.x) {
    // printf("Argmin kernel started with point %lld\n", point);
    float min = FLT_MAX;
    int min_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster) {
      float value = dc[cluster] - 2 * dE[cluster * t + point];
      // printf("C %f E %f (idx %lld) -> Value %f\n", dc[cluster],
      //        dE[cluster * t + point], cluster * t + point, value);
      if (value < min) {
        min = value;
        min_cluster = cluster;
      }
    }

    // convergence-related things
    // printf("Min %f idx %lld\n", min, min_cluster);
    // if (min_cluster != local_assignments[point])
    //   *converged = false;
    atomicAdd(local_k_means_objective_score, min);
    // current objective minus previous objective
    float delta = min - prev_point_to_cluster_distances[point];
    atomicAdd(local_k_means_objective_delta, delta);
    prev_point_to_cluster_distances[point] = min;

    // update assignment and cluster sizes
    local_assignments[point] = min_cluster;
    atomicAdd(&local_cluster_sizes[min_cluster], 1);
    // printf("Finished argmin kernel\n");
  }
}

__global__ void argmin_kernel_simple(int64_t k, int64_t t, float* dE, float* dc, int* local_assignments, int* local_cluster_sizes, cpop::FloatI32 * local_minpairs)
{
  for (int64_t point = blockIdx.x * blockDim.x + threadIdx.x; point < k;
       point += blockDim.x * gridDim.x) {
    local_cluster_sizes[point] = 0;
  }

  for (int64_t point = blockIdx.x * blockDim.x + threadIdx.x; point < t;
       point += blockDim.x * gridDim.x) {
    // printf("Argmin kernel started with point %lld\n", point);
    float min = FLT_MAX;
    int min_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster) {
      float value = dc[cluster] - 2 * dE[cluster * t + point];
      // printf("C %f E %f (idx %lld) -> Value %f\n", dc[cluster],
      //        dE[cluster * t + point], cluster * t + point, value);
      if (value < min) {
        min = value;
        min_cluster = cluster;
      }
    }


    // update assignment and cluster sizes
    local_minpairs[point] = {min, min_cluster};
    atomicAdd(&local_cluster_sizes[min_cluster], 1);
  }
}

__global__ void reinit_kernel2d(float* d_values, int * d_rowinds, int * d_colptrs, int * d_mininds, int * d_cluster_sizes, 
                                int64_t k, int64_t m, int64_t nnz, 
                                cpop::IsMine& op)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += blockDim.x * gridDim.x) 
    {
        d_colptrs[i+1] = (int)op(d_mininds[i]);

    }
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nnz;
         i += blockDim.x * gridDim.x) 
    {
        d_values[i] = 1.0f / d_cluster_sizes[d_rowinds[i]];
        d_rowinds[i] = d_rowinds[i] % k; // map to local rowinds
    }
}

__global__ void reinit_kernel(float* V_global_values, int* global_assignments,
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
      if ((int64_t)(global_assignments[point]) == cluster) {
        V_global_values[i] = 1.0f / global_cluster_sizes[cluster];
      }
    }
  }
}

__global__ void init_from_rowinds_kernel(int * d_rowinds, int * d_cluster_sizes, float * d_vals, int64_t nnz)
{
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nnz; i += blockDim.x * gridDim.x) 
  {
    d_vals[i] = 1.0f/d_cluster_sizes[d_rowinds[i]];
  }
}

__global__ void score_kernel(float* local_scores, float* dK, float* dE,
                             float* dc, int* local_assignments, int64_t t) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < t;
       i += blockDim.x * gridDim.x) {
    int a = local_assignments[i];
    local_scores[i] = dK[t * i + i] - 2 * dE[t * a + i] + dc[a];
  }
}

__global__ void mininds_kernel(cpop::FloatI32 * d_minpairs, int * d_mininds, int cols)
{
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += blockDim.x * gridDim.x)
  {
      d_mininds[i] = d_minpairs[i].i;
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

void launch_z_kernel(int64_t t, float* z, int* assignments, float* ET) {
  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (t + nthreads - 1) / nthreads);

  z_vector_kernel<<<nblocks, nthreads>>>(t, z, assignments, ET);
}

void launch_argmin_kernel(int64_t k, int64_t t, float* dE, float* dc,
                          int* local_assignments, int* local_cluster_sizes,
                          bool* converged, float* local_k_means_objective_score,
                          float* local_k_means_objective_delta,
                          float* prev_point_to_cluster_distances) {
  if (k == 0 || t == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (t + nthreads - 1) / nthreads);
  argmin_kernel<<<nblocks, nthreads>>>(
      k, t, dE, dc, local_assignments, local_cluster_sizes, converged,
      local_k_means_objective_score, local_k_means_objective_delta,
      prev_point_to_cluster_distances);
}


void launch_argmin_kernel_simple(int64_t k, int64_t t, float* dE, float* dc, int* local_assignments, int* local_cluster_sizes, FloatI32 * local_minpairs)
{
  if (k == 0 || t == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (t + nthreads - 1) / nthreads);
  argmin_kernel_simple<<<nblocks, nthreads>>>(
      k, t, dE, dc, local_assignments, local_cluster_sizes, local_minpairs);
}


void launch_reinit_kernel(float* V_global_values, int* global_assignments,
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

void launch_reinit_kernel2d(float * d_values, int * d_rowinds, int * d_colptrs, int * d_mininds, int * d_cluster_sizes, int64_t k, int64_t m, int64_t nnz, IsMine& op)
{
  if (k == 0 || m == 0 || nnz == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (m + nthreads - 1) / nthreads);
  reinit_kernel2d<<<nblocks, nthreads>>>(d_values, 
                                         d_rowinds,
                                         d_colptrs,
                                         d_mininds,
                                         d_cluster_sizes,
                                         k, m, nnz, 
                                         op);
}


void launch_init_from_rowinds_kernel(int * d_rowinds, int * d_colptrs, int * d_cluster_sizes, float * d_vals, int64_t nnz, int64_t m)
{
    if (nnz == 0)
    {
        return;
    }

    // Set the colptrs array
    // TODO: Have option to remove this if it is already initialized
    thrust::fill( thrust::device_pointer_cast(d_colptrs) + 1,
                    thrust::device_pointer_cast(d_colptrs) + m + 1,
                    1);
    cudaDeviceSynchronize();
    launch_inclusive_scan(d_colptrs+1, d_colptrs+1, m);

    // Set values
    int nthreads = 256;
    int nblocks = std::min(int64_t(1048576), (nnz + nthreads - 1) / nthreads);
    init_from_rowinds_kernel<<<nblocks, nthreads>>>(d_rowinds, d_cluster_sizes, d_vals, nnz);

}



void launch_score_kernel(float* local_scores, float* dK, float* dE, float* dc,
                         int* local_assignments, int64_t t) {
  if (t == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (t + nthreads - 1) / nthreads);
  score_kernel<<<nblocks, nthreads>>>(local_scores, dK, dE, dc,
                                      local_assignments, t);
}

void launch_mininds_kernel(FloatI32 * d_minpairs, int * d_mininds, int64_t cols)
{
  if (cols == 0)
    return;

  // 1024 max threads for current CUDA compute capability (<= 7.5)
  // 16x16 blocks, with upwards round for more coverage
  // block cap is set to prevent overflow
  int nthreads = 256;
  int nblocks = std::min(int64_t(1048576), (cols + nthreads - 1) / nthreads);
  mininds_kernel<<<nblocks, nthreads>>>(d_minpairs, d_mininds, cols);
}

void launch_inclusive_scan(int * d_in, int * d_out, int64_t n)
{
  thrust::inclusive_scan(thrust::device_pointer_cast(d_in),
                         thrust::device_pointer_cast(d_in) + n,
                         thrust::device_pointer_cast(d_out));
}

bool test_convergence_equality(int* assignments, int* prev_assignments,
                               int64_t t) {
  thrust::device_ptr<int> t_assignments =
      thrust::device_pointer_cast(assignments);
  thrust::device_ptr<int> t_prev_assignments =
      thrust::device_pointer_cast(prev_assignments);
  return thrust::equal(t_assignments, t_assignments + t, t_prev_assignments);
}

}  // namespace cpop
