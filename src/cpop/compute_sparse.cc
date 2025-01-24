#include <cassert>

#include "cuda_runtime.h"
#include "mpi.h"

#include "compute_sparse.hh"

#include <iostream>

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

namespace cpop {

int Vmat::initialize(cusparseHandle_t& handle, int m, int k, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // TODO: this is a danger for redundancy. The tile sizes of all processes should be stored in a global vector somewhere or passed as an arg.
  int def_t_size = m / size;  // default tile size
  int my_t_size = rank == size - 1 ? m / size + m % size : m / size;

  int* assignments = (int*)calloc(my_t_size, sizeof(int));
  int32_t* cluster_sizes_loc = (int32_t*)calloc(k, sizeof(int32_t));

  int rr_start_cluster = (rank * def_t_size) % k;  // round robin
  for (int i = 0; i < my_t_size; ++i) {
    int cluster = (rr_start_cluster + i) % k;
    assignments[i] = cluster;
    cluster_sizes_loc[cluster]++;
  }

  compute_from_cluster_assignments(handle, m, k, my_t_size, assignments,
                                   cluster_sizes_loc, comm);
  return 0;
}

int Vmat::compute_from_cluster_assignments(cusparseHandle_t& handle, int m,
                                           int k, int t_size, int* assignments,
                                           int32_t* cluster_sizes_loc,
                                           MPI_Comm comm) {
  // TODO: simplify this beast of a method
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Allreduce assignments array over columns
  auto assignments_global = (int*)calloc(m, sizeof(int));
  auto displs = (int*)malloc(size * sizeof(int));
  auto rcounts = (int*)malloc(size * sizeof(int));
  for (int i = 0; i < size; ++i) {
    displs[i] = i * (m / size);
    // Beware that this is dependent on the overall tiling scheme!
    // TODO: this is a danger for redundancy. The tile sizes of all processes should be stored in a global vector somewhere or passed as an arg.
    rcounts[i] = i == size - 1 ? m / size + m % size : m / size;
  }
  MPI_Allgatherv(assignments, t_size, MPI_INT, assignments_global, rcounts,
                 displs, MPI_INT, MPI_COMM_WORLD);

  // Allreduce cluster sizes over columns
  auto cluster_sizes_global = (int32_t*)calloc(k, sizeof(int32_t));
  MPI_Allreduce(cluster_sizes_loc, cluster_sizes_global, k, MPI_INT32_T,
                MPI_SUM, MPI_COMM_WORLD);

  // Manual Global CSR encoding
  int32_t* row_offsets = (int32_t*)calloc(k + 1, sizeof(int32_t));
  // TODO: convert to kernel for speediness (also reduces number of cudaMemcpys)
  for (int i = 0; i < k; i += 1)
    row_offsets[i + 1] = row_offsets[i] + cluster_sizes_global[i];

  int32_t* col_inds = (int32_t*)malloc(m * sizeof(int32_t));
  float* values = (float*)malloc(m * sizeof(float));
  // TODO: convert to kernel for speediness (also reduces number of cudaMemcpys)
  int32_t* cluster_loc_ptrs = (int32_t*)calloc(k, sizeof(int32_t));
  for (int32_t i = 0; i < m; ++i) {
    int cluster = assignments_global[i];  // get the cluster for this point
    int offset = row_offsets[cluster] + (cluster_loc_ptrs[cluster]++);
    values[offset] = 1.0f / cluster_sizes_global[cluster];
    col_inds[offset] = i;
  }
  int32_t* d_row_offsets;
  cudaMalloc(&d_row_offsets, (k + 1) * sizeof(int32_t));
  cudaMemcpy(d_row_offsets, row_offsets, (k + 1) * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  int32_t* d_col_inds;
  cudaMalloc(&d_col_inds, m * sizeof(int32_t));
  cudaMemcpy(d_col_inds, col_inds, m * sizeof(int32_t), cudaMemcpyHostToDevice);
  float* d_values;
  cudaMalloc(&d_values, m * sizeof(float));
  cudaMemcpy(d_values, values, m * sizeof(float), cudaMemcpyHostToDevice);
  cusparseCreateCsr(&global_v, k, m, m, d_row_offsets, d_col_inds, d_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  free(row_offsets);
  free(col_inds);
  free(values);

  // Create local CSC matrix
  int32_t* row_inds = assignments;  // row inds are exactly assignments in CSC
  int32_t* col_offsets = (int32_t*)calloc(t_size + 1, sizeof(int32_t));
  float* csc_values = (float*)calloc(t_size, sizeof(float));
  for (int i = 0; i < t_size; ++i) {
    col_offsets[i + 1] = i + 1;  // this works for CSC since 1 point per column
    csc_values[i] = 1.0f / cluster_sizes_global[assignments[i]];
  }
  int32_t* d_row_inds;
  cudaMalloc(&d_row_inds, t_size * sizeof(int32_t));
  cudaMemcpy(d_row_inds, row_inds, t_size * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  int32_t* d_col_offsets;
  cudaMalloc(&d_col_offsets, (t_size + 1) * sizeof(int32_t));
  cudaMemcpy(d_col_offsets, col_offsets, (t_size + 1) * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  float* d_csc_values;
  cudaMalloc(&d_csc_values, t_size * sizeof(float));
  cudaMemcpy(d_csc_values, csc_values, t_size * sizeof(float),
             cudaMemcpyHostToDevice);
  free(row_inds);  // this frees assignments!
  free(col_offsets);
  free(csc_values);
  cusparseCreateCsc(&local_v, k, t_size, t_size, d_col_offsets, d_row_inds,
                    d_csc_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // free helpers
  free(assignments_global);
  free(displs);
  free(rcounts);
  free(cluster_sizes_global);
  free(cluster_sizes_loc);
  return 0;
}

// cusparseSpMatDescr_t initialize_v(cusparseHandle_t& handle, int n, int k,
//                                   MPI_Comm comm) {
//   int rank;
//   MPI_Comm_rank(comm, &rank);

//   // Manual CSR encoding of the initialized state
//   int32_t* row_offsets = (int32_t*)malloc((k + 1) * sizeof(int32_t));
//   row_offsets[0] = 0;
//   int32_t* col_inds = (int32_t*)malloc(n * sizeof(int32_t));
//   float* values = (float*)malloc(n * sizeof(float));

//   for (int32_t r = 0; r < k; ++r) {
//     int32_t l = (n / k) + ((r < n % k) ? 1 : 0);
//     row_offsets[r + 1] = row_offsets[r] + l;
//     for (int32_t i = 0; i < l; ++i) {
//       col_inds[i + row_offsets[r]] = r + i * k;
//       values[i + row_offsets[r]] = 1.0f / l;
//     }
//   }

//   int32_t* d_row_offsets;
//   cudaMalloc(&d_row_offsets, (k + 1) * sizeof(int32_t));
//   cudaMemcpy(d_row_offsets, row_offsets, (k + 1) * sizeof(int32_t),
//              cudaMemcpyHostToDevice);
//   int32_t* d_col_inds;
//   cudaMalloc(&d_col_inds, n * sizeof(int32_t));
//   cudaMemcpy(d_col_inds, col_inds, n * sizeof(int32_t), cudaMemcpyHostToDevice);
//   float* d_values;
//   cudaMalloc(&d_values, n * sizeof(float));
//   cudaMemcpy(d_values, values, n * sizeof(float), cudaMemcpyHostToDevice);

//   free(row_offsets);
//   free(col_inds);
//   free(values);

//   cusparseSpMatDescr_t V;
//   cusparseCreateCsr(&V, k, n, n, d_row_offsets, d_col_inds, d_values,
//                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
//   return V;
// }

// cusparseSpMatDescr_t initialize_local_v(cusparseHandle_t& handle, int m, int k,
//                                         MPI_Comm comm) {}

cusparseDnMatDescr_t spmm(cusparseHandle_t& handle, cusparseSpMatDescr_t& V,
                          cusparseDnMatDescr_t& K) {
  // Get input information
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  cudaDataType type;
  cusparseOrder_t order;
  float* dn_values;
  cusparseSpMatGetSize(V, &sp_rows, &sp_cols, &nnz);
  cusparseDnMatGet(K, &dn_rows, &dn_cols, &ld, (void**)&dn_values, &type,
                   &order);

  assert(sp_cols == dn_rows && "Inner dimension must be equal in size.");
  assert(type == CUDA_R_32F && "Matrix data must be FP32.");

  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

  // Allocate memory for output
  float* values;
  cudaMalloc(&values, sp_rows * dn_cols * sizeof(float));
  cusparseDnMatDescr_t ET;
  cusparseCreateDnMat(&ET, sp_rows, dn_cols, dn_cols, (void*)values, type,
                      order);

  // Allocate workspace buffer
  size_t buffer_size;
  void* buffer;
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K, &beta,
                          ET, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &buffer_size);
  cudaMalloc(&buffer, buffer_size);

  // Preprocess
  cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K, &beta,
                          ET, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  // Perform SpMM
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K, &beta, ET,
               CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  // Clean up
  cudaFree(buffer);

  return ET;
}

}  // namespace cpop
