#include <cassert>

#include "cuda_runtime.h"
#include "mpi.h"

#include "compute_sparse.hh"

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

cusparseSpMatDescr_t initialize_v(cusparseHandle_t& handle, int n, int k,
                                  MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  // Manual CSR encoding of the initialized state
  int32_t* row_offsets = (int32_t*)malloc((k + 1) * sizeof(int32_t));
  row_offsets[0] = 0;
  int32_t* col_inds = (int32_t*)malloc(n * sizeof(int32_t));
  float* values = (float*)malloc(n * sizeof(float));

  for (int32_t r = 0; r < k; ++r) {
    int32_t l = (n / k) + ((r < n % k) ? 1 : 0);
    row_offsets[r + 1] = row_offsets[r] + l;
    for (int32_t i = 0; i < l; ++i) {
      col_inds[i + row_offsets[r]] = r + i * k;
      values[i + row_offsets[r]] = 1.0f / l;
    }
  }

  int32_t* d_row_offsets;
  cudaMalloc(&d_row_offsets, (k + 1) * sizeof(int32_t));
  cudaMemcpy(d_row_offsets, row_offsets, (k + 1) * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  int32_t* d_col_inds;
  cudaMalloc(&d_col_inds, n * sizeof(int32_t));
  cudaMemcpy(d_col_inds, col_inds, n * sizeof(int32_t), cudaMemcpyHostToDevice);
  float* d_values;
  cudaMalloc(&d_values, n * sizeof(float));
  cudaMemcpy(d_values, values, n * sizeof(float), cudaMemcpyHostToDevice);

  free(row_offsets);
  free(col_inds);
  free(values);

  cusparseSpMatDescr_t V;
  cusparseCreateCsr(&V, k, n, n, d_row_offsets, d_col_inds, d_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  return V;
}

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
