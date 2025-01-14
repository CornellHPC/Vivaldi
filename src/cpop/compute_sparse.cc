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

cusparseSpMatDescr_t initialize_v(cusparseHandle_t& handle, int n, int k) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int32_t* rows = (int32_t*)malloc(n * sizeof(int32_t));
  int32_t* cols = (int32_t*)malloc(n * sizeof(int32_t));
  float* vals = (float*)malloc(n * sizeof(float));

  int idx = 0;
  for (int32_t r = 0; r < k; ++r) {
    int32_t l = (n / k) + ((r < n % k) ? 1 : 0);
    for (int32_t c = r; c < n; c += k) {
      rows[idx] = r;
      cols[idx] = c;
      vals[idx] = 1.0f / l;
      idx++;
    }
  }

  int32_t *cooRowInds, *cooColInds;
  float* cooValues;

  cudaMalloc(&cooRowInds, n * sizeof(int32_t));
  cudaMalloc(&cooColInds, n * sizeof(int32_t));
  cudaMalloc(&cooValues, n * sizeof(float));

  cudaMemcpy(cooRowInds, rows, n * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cooColInds, cols, n * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cooValues, vals, n * sizeof(float), cudaMemcpyHostToDevice);

  free(rows);
  free(cols);
  free(vals);

  cusparseSpMatDescr_t V;
  cusparseCreateCoo(&V, k, n, n, cooRowInds, cooColInds, cooValues,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
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
                          ET, CUDA_R_32F, CUSPARSE_SPMM_COO_ALG4, &buffer_size);
  cudaMalloc(&buffer, buffer_size);

  // Perform SpMM
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K, &beta, ET,
               CUDA_R_32F, CUSPARSE_SPMM_COO_ALG4, buffer);

  // Clean up
  cudaFree(buffer);

  return ET;
}

}  // namespace cpop
