#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cuda_runtime.h"
#include "mpi.h"
#include "mpio.h"

#include "cluster.hh"
#include "gpu_kernels.cuh"
#include "utils.hh"

namespace cpop {

V_t::V_t(int64_t m, int64_t k, bool sparse, MPI_Comm comm) {
  // Struct member initialization
  m_ = m;
  k_ = k;
  this->sparse = sparse;
  this->comm = comm;

  // MPI initializations
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_procs);
  t_sizes = compute_tile_sizes(m, n_procs);
  t = t_sizes[rank];

  displs = (int*)calloc(n_procs, sizeof(int));
  for (int i = 1; i < n_procs; ++i)
    displs[i] = displs[i - 1] + t_sizes[i - 1];  // MPI displacements

  // Struct data initialization
  CHECK_CUDA(cudaMalloc(&global_assignments, m * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&global_cluster_sizes, k * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&local_assignments, t * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&local_cluster_sizes, k * sizeof(int)));

  // round robin initialization (todo: GPU)
  int* init_global_cluster_sizes = (int*)calloc(k, sizeof(int));
  for (int i = 0; i < k; ++i)
    init_global_cluster_sizes[i] = (m / k) + ((i < m % k) ? 1 : 0);
  CHECK_CUDA(cudaMemcpy(global_cluster_sizes, init_global_cluster_sizes,
                        k * sizeof(int), cudaMemcpyHostToDevice));

  // round robin initialization (todo: GPU)
  int* init_assignments = (int*)calloc(m, sizeof(int));
  for (int i = 0; i < m; ++i)
    init_assignments[i] = i % k;
  CHECK_CUDA(cudaMemcpy(global_assignments, init_assignments, m * sizeof(int),
                        cudaMemcpyHostToDevice));

  // set assignment pointers
  local_ptr_to_assignments = global_assignments + displs[rank];
  previous_global_assignments = nullptr;
  CHECK_CUDA(cudaMalloc(&converged, sizeof(bool)));

  // Implementation-specific initialization
  if (sparse) {
    CHECK_CUDA(cudaMalloc(&values, m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&global_csc_col_offsets, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&local_csc_col_offsets, (t + 1) * sizeof(int)));

    // basic CSC initializations (todo: GPU)
    int* global_csc_col_offsets_ = (int*)calloc(m + 1, sizeof(int));
    for (int i = 0; i < m; ++i)
      global_csc_col_offsets_[i + 1] = i + 1;
    CHECK_CUDA(cudaMemcpy(global_csc_col_offsets, global_csc_col_offsets_,
                          (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(local_csc_col_offsets, global_csc_col_offsets_,
                          (t + 1) * sizeof(int), cudaMemcpyHostToDevice));
    free(global_csc_col_offsets_);

    // round robin initialization (todo: GPU)
    float* init_values = (float*)calloc(m, sizeof(float));
    for (int i = 0; i < m; ++i)
      init_values[i] = 1.0f / init_global_cluster_sizes[i % k];
    CHECK_CUDA(cudaMemcpy(values, init_values, m * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(init_values);

    // cusparse initializations
    CHECK_CUSPARSE(cusparseCreateCsc(&gV, k, m, m, global_csc_col_offsets,
                                     global_assignments, values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // the local partition of V in CSC is found by slicing the global partition at [displs[rank]:displs[rank] + t]
    // (since displs[rank] is the displacement of this rank's first point)
    // this is done with simple pointer arithmetic
    local_ptr_to_values = values + displs[rank];
    CHECK_CUSPARSE(cusparseCreateCsc(
        &lV, k, t, t, local_csc_col_offsets, local_ptr_to_assignments,
        local_ptr_to_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  } else {
    // round robin initialization (TODO: gpu)
    float* values = (float*)calloc(k * m, sizeof(float));
    for (int64_t cluster = 0; cluster < k; ++cluster) {
      for (int64_t point = cluster; point < m; point += k) {
        int64_t total = m / k + ((cluster < m % k) ? 1 : 0);
        values[cluster * m + point] = 1.0f / total;
      }
    }
    CHECK_CUDA(cudaMalloc(&this->values, k * m * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(this->values, values, k * m * sizeof(float),
                          cudaMemcpyHostToDevice));
    local_ptr_to_values = this->values + displs[rank];
    free(values);
  }

  // Clean up
  free(init_global_cluster_sizes);
  free(init_assignments);
}

int V_t::save(const char* path) {
  MPI_File fh;
  MPI_File_open(comm, path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                &fh);

  // Compute offset for rank
  int offset = displs[rank];

  int* assignments = (int*)malloc(t * sizeof(int));
  CHECK_CUDA(cudaMemcpy(assignments, local_ptr_to_assignments, t * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Write the data to disk
  MPI_File_write_at(fh, offset * sizeof(int), assignments, t, MPI_INT,
                    MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  free(assignments);
  return EXIT_SUCCESS;
}

void V_t::print() {
  std::cout << "Printing V from rank " << rank << std::endl;

  if (sparse) {
    std::cout << "Local displacement is " << displs[rank] << std::endl;
    std::cout << "Global device vectors:" << std::endl;
    print_device_buffer(global_assignments, m_);
    print_device_buffer(global_csc_col_offsets, m_ + 1);
    print_device_buffer(values, m_);
    std::cout << "Local device vectors:" << std::endl;
    print_device_buffer(local_ptr_to_assignments, t);
    print_device_buffer(local_csc_col_offsets, t + 1);
    print_device_buffer(local_ptr_to_values, t);
    std::cout << "Working vectors:" << std::endl;
    print_device_buffer(global_cluster_sizes, k_);
    print_device_buffer(local_assignments, t);
    print_device_buffer(local_cluster_sizes, k_);
  } else {
    print_device_matrix(values, k_, m_);
  }

  std::cout << "-------------------" << std::endl;
}

bool V_t::test_convergence() {
  if (previous_global_assignments &&
      test_convergence_equality(global_assignments, previous_global_assignments,
                                m_)) {
    // Free everything and return true because we have converged
    CHECK_CUDA(cudaFree(previous_global_assignments));
    previous_global_assignments = nullptr;
    return true;  // Converged
  } else if (!previous_global_assignments) {
    // First iteration, so we need to allocate previous_global_assignments
    CHECK_CUDA(cudaMalloc(&previous_global_assignments, m_ * sizeof(int)));
  }
  // Set previous_global_assignments to current global_assignments
  cudaMemcpy(previous_global_assignments, global_assignments, m_ * sizeof(int),
             cudaMemcpyDeviceToDevice);
  return false;  // Not converged
}

V_t::~V_t() {
  CHECK_CUDA(cudaFree(converged));
  free(t_sizes);
  free(displs);
  if (sparse) {
    CHECK_CUDA(cudaFree(global_assignments));
    CHECK_CUDA(cudaFree(global_cluster_sizes));
    CHECK_CUDA(cudaFree(local_assignments));
    CHECK_CUDA(cudaFree(local_cluster_sizes));
    CHECK_CUDA(cudaFree(values));
    CHECK_CUDA(cudaFree(global_csc_col_offsets));
    CHECK_CUDA(cudaFree(local_csc_col_offsets));
    CHECK_CUSPARSE(cusparseDestroySpMat(gV));
    CHECK_CUSPARSE(cusparseDestroySpMat(lV));
    if (previous_global_assignments)
      CHECK_CUDA(cudaFree(previous_global_assignments));
  } else {
    CHECK_CUDA(cudaFree(values));
  }
}

DnMat_t::DnMat_t(int64_t h, int64_t w) {
  CHECK_CUDA(cudaMalloc(&dM, h * w * sizeof(float)));
  CHECK_CUSPARSE(
      cusparseCreateDnMat(&M, h, w, w, dM, CUDA_R_32F, CUSPARSE_ORDER_ROW));

  h_ = h;
  w_ = w;
}

DnMat_t::DnMat_t(int64_t h, int64_t w, float* dM_) {
  dM = dM_;
  CHECK_CUSPARSE(
      cusparseCreateDnMat(&M, h, w, w, dM, CUDA_R_32F, CUSPARSE_ORDER_ROW));

  h_ = h;
  w_ = w;
}

int DnMat_t::print() {
  print_device_matrix(dM, h_, w_);
  return EXIT_SUCCESS;
}

DnMat_t::~DnMat_t() {
  CHECK_CUDA(cudaFree(dM));
  CHECK_CUSPARSE(cusparseDestroyDnMat(M));
}

DnVec_t::DnVec_t(int t) {
  CHECK_CUDA(cudaMalloc(&dz, t * sizeof(float)));
  CHECK_CUSPARSE(cusparseCreateDnVec(&z, t, dz, CUDA_R_32F));
  size = t;
}

int DnVec_t::print() {
  print_device_buffer(dz, size);
  return EXIT_SUCCESS;
}

DnVec_t::~DnVec_t() {
  CHECK_CUDA(cudaFree(dz));
  CHECK_CUSPARSE(cusparseDestroyDnVec(z));
}

int spmm(Handle& handle, V_t& V, DnMat_t& K, DnMat_t& E) {
  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

  if (handle.isSparse()) {
    // Allocate workspace buffer
    size_t buffer_size;
    void* buffer;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV, K.M, &beta, E.M,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
    CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

    // Perform SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV,
                                K.M, &beta, E.M, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, buffer));

    // Clean up
    CHECK_CUDA(cudaFree(buffer));
  } else {
    int64_t t = V.t;
    int64_t m = V.m_;
    int64_t k = V.k_;
    CHECK_CUBLAS(cublasSgemm(handle.dh(), CUBLAS_OP_N, CUBLAS_OP_N, t, k, m,
                             &alpha, K.dM, t, V.values, m, &beta, E.dM, t));
  }

  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

int compute_z(V_t& V, DnMat_t& E, DnVec_t& z) {
  // Because we're using CSC, the cluster assignments vector is exactly the
  // CSC row indices vector of V
  launch_z_kernel(z.size, z.dz, V.local_ptr_to_assignments, E.dM);
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

int spmv(Handle& handle, V_t& V, DnVec_t& z, DnVec_t& c) {
  float alpha = 1.0f;
  float beta = 0.0f;

  if (handle.isSparse()) {
    // allocate an external buffer if needed
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.lV, z.z, &beta,
        c.z, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(handle.sh(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, V.lV, z.z, &beta, c.z, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // cleanup
    CHECK_CUDA(cudaFree(dBuffer));
  } else {
    int64_t t = V.t;
    int64_t k = V.k_;
    int64_t m = V.m_;
    CHECK_CUBLAS(cublasSgemv(handle.dh(), CUBLAS_OP_T, t, k, &alpha,
                             V.local_ptr_to_values, m, z.dz, 1, &beta, c.dz,
                             1));
  }

  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

int sum_vec(DnVec_t& c, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, c.dz, c.size, MPI_FLOAT, MPI_SUM, comm);
  return EXIT_SUCCESS;
}

int compute_c(Handle& handle, V_t& V, DnVec_t& z, DnVec_t& c, MPI_Comm comm) {
  spmv(handle, V, z, c);  // SpMV: c = Vz using local V
  sum_vec(c, comm);       // Calculate global c by summing across ranks
  return EXIT_SUCCESS;
}

int argmin(DnMat_t& E, DnVec_t& c, V_t& V) {
  bool t = true;
  cudaMemcpy(V.converged, &t, sizeof(bool), cudaMemcpyHostToDevice);
  launch_argmin_kernel(V.k_, V.t, E.dM, c.dz, V.local_assignments,
                       V.local_cluster_sizes, V.converged);
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

int gather_assignments(DnMat_t& E, DnVec_t& c, V_t& V, int convergence) {
  int send_count = V.t;
  int* send_buffer = V.local_assignments;
  int* recv_sizes = V.t_sizes;

  int dead_process_count = 0;
  if (convergence) {
    if (convergence == 2)
      recv_sizes = (int*)malloc(V.n_procs * sizeof(int));

    bool locally_converged;
    cudaMemcpy(&locally_converged, V.converged, sizeof(bool),
               cudaMemcpyDeviceToHost);
    if (convergence == 2 && locally_converged) {
      // this process has locally converged, so it can be removed from allgather
      send_count = 0;
      send_buffer = nullptr;
    }

    bool local_convergence_ptr[V.n_procs];
    MPI_Allgather(&locally_converged, 1, MPI_C_BOOL, local_convergence_ptr, 1,
                  MPI_C_BOOL, V.comm);
    for (int i = 0; i < V.n_procs; ++i) {
      if (local_convergence_ptr[i])
        dead_process_count++;
      if (convergence == 2) {
        // exclude relevant processes from allgather
        if (local_convergence_ptr[i]) {
          recv_sizes[i] = 0;
        } else {
          recv_sizes[i] = V.t_sizes[i];
        }
      }
    }
    if (dead_process_count == V.n_procs)
      return dead_process_count;  // all processes are dead, quit
  }

  // this one always utilizes all processes, but only passes k-size vector
  MPI_Allreduce(V.local_cluster_sizes, V.global_cluster_sizes, V.k_, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  // this one will be reduced to the relevant processes as it allgathers a large
  // dense vector of points (if process-exlusion-based-convergence is enabled)
  MPI_Allgatherv(send_buffer, send_count, MPI_INT, V.global_assignments,
                 recv_sizes, V.displs, MPI_INT, MPI_COMM_WORLD);

  if (convergence == 2)
    free(recv_sizes);
  return dead_process_count;
}

int set_V_from_assignments(DnMat_t& E, DnVec_t& c, V_t& V) {
  launch_reinit_kernel(V.values, V.global_assignments, V.global_cluster_sizes,
                       V.k_, V.m_, V.sparse);
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

int reinit_V(DnMat_t& E, DnVec_t& c, V_t& V) {
  argmin(E, c, V);                     // Launch argmin kernel
  gather_assignments(E, c, V, false);  // Gather assignments and cluster sizes
  set_V_from_assignments(E, c, V);     // Launch reinit kernel
  return EXIT_SUCCESS;
}

}  // namespace cpop
