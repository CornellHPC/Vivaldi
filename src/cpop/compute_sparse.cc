#include <cassert>

#include "cuda_runtime.h"
#include "mpi.h"

#include "compute_sparse.hh"
#include "cusparse_helpers.hh"
#include "utils.hh"

namespace cpop {

int64_t* compute_g_assignments(int64_t m, int64_t t, int64_t* assignments,
                               int n_procs, int* t_sizes) {
  auto g_assignments = (int64_t*)calloc(m, sizeof(int64_t));
  auto displs = (int*)calloc(n_procs, sizeof(int));
  for (int i = 1; i < n_procs; ++i) {
    displs[i] = displs[i - 1] + t_sizes[i - 1];
  }
  MPI_Allgatherv(assignments, t, MPI_INT64_T, g_assignments, t_sizes, displs,
                 MPI_INT64_T, MPI_COMM_WORLD);
  return g_assignments;
}

int64_t* compute_g_cluster_sizes(int64_t k, int64_t* cluster_sizes) {
  auto g_cluster_sizes = (int64_t*)calloc(k, sizeof(int64_t));
  MPI_Allreduce(cluster_sizes, g_cluster_sizes, k, MPI_INT64_T, MPI_SUM,
                MPI_COMM_WORLD);
  return g_cluster_sizes;
}

int create_gV_csr(cusparseSpMatDescr_t* gV, int64_t m, int64_t k,
                  int64_t* g_assignments, int64_t* g_cluster_sizes) {
  int64_t* row_offsets = (int64_t*)calloc(k + 1, sizeof(int64_t));
  // TODO: convert to kernel for speediness (also reduces number of cudaMemcpys)
  for (int i = 0; i < k; i += 1)
    row_offsets[i + 1] = row_offsets[i] + g_cluster_sizes[i];
  int64_t* col_inds = (int64_t*)malloc(m * sizeof(int64_t));
  float* values = (float*)malloc(m * sizeof(float));
  // TODO: convert to kernel for speediness (also reduces number of cudaMemcpys)
  int64_t* cluster_loc_ptrs = (int64_t*)calloc(k, sizeof(int64_t));
  for (int64_t i = 0; i < m; ++i) {
    int cluster = g_assignments[i];  // get the cluster for this point
    int offset = row_offsets[cluster] + (cluster_loc_ptrs[cluster]++);
    values[offset] = 1.0f / g_cluster_sizes[cluster];
    col_inds[offset] = i;
  }
  free(cluster_loc_ptrs);
  int64_t* d_row_offsets;
  CHECK_CUDA(cudaMalloc(&d_row_offsets, (k + 1) * sizeof(int64_t)));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, row_offsets, (k + 1) * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  int64_t* d_col_inds;
  CHECK_CUDA(cudaMalloc(&d_col_inds, m * sizeof(int64_t)));
  CHECK_CUDA(cudaMemcpy(d_col_inds, col_inds, m * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  float* d_values;
  CHECK_CUDA(cudaMalloc(&d_values, m * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_values, values, m * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUSPARSE(cusparseCreateCsr(
      gV, k, m, m, d_row_offsets, d_col_inds, d_values, CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  free(row_offsets);
  free(col_inds);
  free(values);
  return 0;
}

int create_lV_csc(cusparseSpMatDescr_t* lV, int64_t t, int64_t k,
                  int64_t* assignments, int64_t* g_cluster_sizes) {
  int64_t* row_inds = assignments;  // row inds are exactly assignments in CSC
  int64_t* col_offsets = (int64_t*)calloc(t + 1, sizeof(int64_t));
  float* csc_values = (float*)calloc(t, sizeof(float));
  for (int i = 0; i < t; ++i) {
    col_offsets[i + 1] = i + 1;  // this works for CSC since 1 point per column
    csc_values[i] = 1.0f / g_cluster_sizes[assignments[i]];
  }
  int64_t* d_row_inds;
  CHECK_CUDA(cudaMalloc(&d_row_inds, t * sizeof(int64_t)));
  CHECK_CUDA(cudaMemcpy(d_row_inds, row_inds, t * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  int64_t* d_col_offsets;
  CHECK_CUDA(cudaMalloc(&d_col_offsets, (t + 1) * sizeof(int64_t)));
  CHECK_CUDA(cudaMemcpy(d_col_offsets, col_offsets, (t + 1) * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  float* d_csc_values;
  CHECK_CUDA(cudaMalloc(&d_csc_values, t * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_csc_values, csc_values, t * sizeof(float),
                        cudaMemcpyHostToDevice));
  // don't free row_inds since this is assignments which was passed as arg
  free(col_offsets);
  free(csc_values);
  CHECK_CUSPARSE(cusparseCreateCsc(
      lV, k, t, t, d_col_offsets, d_row_inds, d_csc_values, CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  return 0;
}

int reinit_V(cusparseSpMatDescr_t* gV, cusparseSpMatDescr_t* lV, int64_t m,
             int64_t t, int64_t k, int64_t* assignments, int64_t* cluster_sizes,
             int* t_sizes, MPI_Comm comm) {
  int n_procs;
  MPI_Comm_size(comm, &n_procs);

  // These two functions are essentially MPI directives that make it so that every process
  // knows the final global assignments for each point (based on the distributed D matrix)
  int64_t* g_assignments =
      compute_g_assignments(m, t, assignments, n_procs, t_sizes);
  int64_t* g_cluster_sizes = compute_g_cluster_sizes(k, cluster_sizes);

  create_gV_csr(gV, m, k, g_assignments, g_cluster_sizes);
  create_lV_csc(lV, t, k, assignments, g_cluster_sizes);
  return 0;
}

int init_V(cusparseSpMatDescr_t* gV, cusparseSpMatDescr_t* lV, int64_t m,
           int64_t t, int64_t k, int* t_sizes, MPI_Comm comm) {
  int rank, n_procs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_procs);

  int64_t* assignments = (int64_t*)calloc(t, sizeof(int64_t));
  int64_t* cluster_sizes = (int64_t*)calloc(k, sizeof(int64_t));

  int rr_start_cluster = 0;  // round robin
  for (int i = 0; i < rank; ++i)
    rr_start_cluster += t_sizes[i];
  rr_start_cluster %= k;

  for (int i = 0; i < t; ++i) {
    int cluster = (rr_start_cluster + i) % k;
    assignments[i] = cluster;
    cluster_sizes[cluster]++;
  }

  return reinit_V(gV, lV, m, t, k, assignments, cluster_sizes, t_sizes, comm);
}

int spmm(cusparseHandle_t& handle, cusparseSpMatDescr_t& V,
         cusparseDnMatDescr_t& K, cusparseDnMatDescr_t* ET) {
  // Get input information
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  cudaDataType type;
  cusparseOrder_t order;
  float* dn_values;
  CHECK_CUSPARSE(cusparseSpMatGetSize(V, &sp_rows, &sp_cols, &nnz));
  CHECK_CUSPARSE(cusparseDnMatGet(K, &dn_rows, &dn_cols, &ld,
                                  (void**)&dn_values, &type, &order));

  assert(sp_cols == dn_rows && "Inner dimension must be equal in size.");
  assert(type == CUDA_R_32F && "Matrix data must be FP32.");

  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

  // Allocate memory for output
  float* values;
  CHECK_CUDA(cudaMalloc(&values, sp_rows * dn_cols * sizeof(float)));
  CHECK_CUSPARSE(cusparseCreateDnMat(ET, sp_rows, dn_cols, dn_cols,
                                     (void*)values, type, order));

  // Allocate workspace buffer
  size_t buffer_size;
  void* buffer;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K, &beta, *ET, CUDA_R_32F,
      CUSPARSE_SPMM_CSR_ALG2, &buffer_size));
  CHECK_CUDA(cudaMalloc(&buffer, buffer_size));

  // // Preprocess (may not work with later cusparseSpMV_preprocess)
  // cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K, &beta,
  //                         *ET, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  // Perform SpMM
  CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K,
                              &beta, *ET, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                              buffer));

  // Clean up
  // CHECK_CUDA(cudaFree(buffer));

  return 0;
}

}  // namespace cpop
