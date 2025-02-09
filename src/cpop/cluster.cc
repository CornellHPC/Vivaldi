#include <cassert>
#include <cstring>
#include <iostream>

#include "cuda_runtime.h"
#include "mpi.h"
#include "mpio.h"

#include "cluster.hh"
#include "gpu_kernels.cuh"
#include "utils.hh"

void print_cu_vec(int64_t* vec, int64_t size) {
  int64_t* h_vec = (int64_t*)malloc(size * sizeof(int64_t));
  cudaMemcpy(h_vec, vec, size * sizeof(int64_t), cudaMemcpyDeviceToHost);
  std::cout << "Size " << size << "vector [ ";
  for (int i = 0; i < size; ++i) {
    std::cout << h_vec[i] << " ";
  }
  std::cout << "]" << std::endl;
  free(h_vec);
}

void print_cu_vec(int* vec, int64_t size) {
  int* h_vec = (int*)malloc(size * sizeof(int));
  cudaMemcpy(h_vec, vec, size * sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Size " << size << "vector [ ";
  for (int i = 0; i < size; ++i) {
    std::cout << h_vec[i] << " ";
  }
  std::cout << "]" << std::endl;
  free(h_vec);
}

void print_cu_vec(float* vec, int64_t size) {
  float* h_vec = (float*)malloc(size * sizeof(float));
  cudaMemcpy(h_vec, vec, size * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Size " << size << "vector [ ";
  for (int i = 0; i < size; ++i) {
    std::cout << h_vec[i] << " ";
  }
  std::cout << "]" << std::endl;
  free(h_vec);
}

namespace cpop {

V_t::V_t(int64_t m, int64_t t, int64_t k, int* t_sizes, MPI_Comm comm) {
  // CUDA array initializations
  CHECK_CUDA(cudaMalloc(&global_assignments, m * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&global_cluster_sizes, k * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&local_assignments, t * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&local_cluster_sizes, k * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&values, m * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&global_csc_col_offsets, (m + 1) * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&local_csc_col_offsets, (t + 1) * sizeof(int64_t)));

  // MPI initializations
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_procs);
  t_sizes_ = t_sizes;
  displs = (int*)calloc(n_procs, sizeof(int));
  for (int i = 1; i < n_procs; ++i)
    displs[i] = displs[i - 1] + t_sizes_[i - 1];  // MPI displacements

  // basic CSC initializations (todo: GPU)
  int64_t* global_csc_col_offsets_ = (int64_t*)calloc(m + 1, sizeof(int64_t));
  for (int64_t i = 0; i < m; ++i)
    global_csc_col_offsets_[i + 1] = i + 1;
  CHECK_CUDA(cudaMemcpy(global_csc_col_offsets, global_csc_col_offsets_,
                        (m + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(local_csc_col_offsets, global_csc_col_offsets_,
                        (t + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  free(global_csc_col_offsets_);

  // round robin initialization (todo: GPU)
  int* init_global_cluster_sizes = (int*)calloc(k, sizeof(int));
  int64_t* init_assignments = (int64_t*)calloc(m, sizeof(int64_t));
  float* init_values = (float*)calloc(m, sizeof(float));
  for (int64_t i = 0; i < k; ++i)
    init_global_cluster_sizes[i] = (m / k) + ((i < m % k) ? 1 : 0);
  CHECK_CUDA(cudaMemcpy(global_cluster_sizes, init_global_cluster_sizes,
                        k * sizeof(int), cudaMemcpyHostToDevice));
  for (int64_t i = 0; i < m; ++i) {
    init_assignments[i] = i % k;
    init_values[i] = 1.0f / init_global_cluster_sizes[i % k];
  }
  CHECK_CUDA(cudaMemcpy(global_assignments, init_assignments,
                        m * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(values, init_values, m * sizeof(float),
                        cudaMemcpyHostToDevice));
  free(init_global_cluster_sizes);
  free(init_assignments);
  free(init_values);

  // cusparse initializations
  CHECK_CUSPARSE(cusparseCreateCsc(&gV, k, m, m, global_csc_col_offsets,
                                   global_assignments, values,
                                   CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // the local partition of V in CSC is found by slicing the global partition at [displs[rank]:displs[rank] + t]
  // (since displs[rank] is the displacement of this rank's first point)
  // this is done with simple pointer arithmetic
  local_ptr_to_assignments = global_assignments + displs[rank];
  local_ptr_to_values = values + displs[rank];
  CHECK_CUSPARSE(cusparseCreateCsc(
      &lV, k, t, t, local_csc_col_offsets, local_ptr_to_assignments,
      local_ptr_to_values, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  m_ = m;
  t_ = t;
  k_ = k;
  this->comm = comm;
}

int V_t::reinit(float* dE, float* dc) {
  // compute argmin
  launch_argmin_kernel(k_, t_, dE, dc, local_assignments, local_cluster_sizes);

  // reduce over nprocs
  MPI_Allreduce(local_cluster_sizes, global_cluster_sizes, k_, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allgatherv(local_assignments, t_, MPI_INT64_T, global_assignments,
                 t_sizes_, displs, MPI_INT64_T, MPI_COMM_WORLD);

  // reinitialize
  launch_reinit_kernel(values, global_assignments, global_cluster_sizes, m_);
  return EXIT_SUCCESS;
}

int V_t::save(const char* path) {
  MPI_File fh;
  MPI_File_open(comm, path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                &fh);

  // Compute offset for rank
  int offset = displs[rank];

  int64_t* assignments = (int64_t*)malloc(t_ * sizeof(int64_t));
  CHECK_CUDA(cudaMemcpy(assignments, local_ptr_to_assignments,
                        t_ * sizeof(int64_t), cudaMemcpyDeviceToHost));
  std::cout << "Final assignments [ ";
  for (int i = 0; i < t_; ++i) {
    std::cout << assignments[i] << " ";
  }
  std::cout << "]" << std::endl;

  // Write the data to disk
  MPI_File_write_at(fh, offset * sizeof(int64_t), assignments, t_, MPI_INT64_T,
                    MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  free(assignments);
  return EXIT_SUCCESS;
}

void V_t::print() {
  std::cout << "Printing V from rank " << rank << std::endl;
  std::cout << "Local displacement is " << displs[rank] << std::endl;
  std::cout << "Global device vectors:" << std::endl;
  print_cu_vec(global_assignments, m_);
  print_cu_vec(global_csc_col_offsets, m_ + 1);
  print_cu_vec(values, m_);
  std::cout << "Local device vectors:" << std::endl;
  print_cu_vec(local_ptr_to_assignments, t_);
  print_cu_vec(local_csc_col_offsets, t_ + 1);
  print_cu_vec(local_ptr_to_values, t_);
  std::cout << "Working vectors:" << std::endl;
  print_cu_vec(global_cluster_sizes, k_);
  print_cu_vec(local_assignments, t_);
  print_cu_vec(local_cluster_sizes, k_);
  std::cout << "-------------------" << std::endl;
}

V_t::~V_t() {
  CHECK_CUDA(cudaFree(global_assignments));
  CHECK_CUDA(cudaFree(global_cluster_sizes));
  CHECK_CUDA(cudaFree(local_assignments));
  CHECK_CUDA(cudaFree(local_cluster_sizes));
  CHECK_CUDA(cudaFree(values));
  CHECK_CUDA(cudaFree(global_csc_col_offsets));
  CHECK_CUDA(cudaFree(local_csc_col_offsets));
  CHECK_CUSPARSE(cusparseDestroySpMat(gV));
  CHECK_CUSPARSE(cusparseDestroySpMat(lV));
  free(displs);
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
  float* m = (float*)malloc(h_ * w_ * sizeof(float));
  CHECK_CUDA(
      cudaMemcpy(m, dM, h_ * w_ * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < h_; ++i) {
    for (int j = 0; j < w_; ++j) {
      std::cout << m[i * w_ + j] << " ";
    }
    std::cout << std::endl;
  }

  free(m);
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
  float* v = (float*)malloc(size * sizeof(float));
  CHECK_CUDA(cudaMemcpy(v, dz, size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;

  free(v);
  return EXIT_SUCCESS;
}

DnVec_t::~DnVec_t() {
  CHECK_CUDA(cudaFree(dz));
  CHECK_CUSPARSE(cusparseDestroyDnVec(z));
}

int spmm(cusparseHandle_t& handle, V_t& V, DnMat_t& K, DnMat_t& E) {
  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

  // Allocate workspace buffer
  size_t buffer_size;
  void* buffer;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV, K.M, &beta, E.M,
      CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
  CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

  // // Preprocess (may not work with later cusparseSpMV_preprocess)
  // cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, gV, K, &beta,
  //                         *ET, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  // Perform SpMM
  CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.gV,
                              K.M, &beta, E.M, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, buffer));

  // Clean up
  CHECK_CUDA(cudaFree(buffer));
  return EXIT_SUCCESS;
}

int compute_z(V_t& V, DnMat_t& E, DnVec_t& z) {
  // Because we're using CSC, the cluster assignments vector is exactly the
  // CSC row indices vector of V
  launch_z_kernel(z.size, z.dz, V.local_ptr_to_assignments, E.dM);
  return EXIT_SUCCESS;
}

int spmv(cusparseHandle_t& handle, V_t& V, DnVec_t& z, DnVec_t& c) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // allocate an external buffer if needed
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V.lV, z.z, &beta, c.z,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // execute SpMV
  CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              V.lV, z.z, &beta, c.z, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

  // cleanup
  CHECK_CUDA(cudaFree(dBuffer));
  return EXIT_SUCCESS;
}

int sum_vec(DnVec_t& c, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, c.dz, c.size, MPI_FLOAT, MPI_SUM, comm);
  return EXIT_SUCCESS;
}

}  // namespace cpop
