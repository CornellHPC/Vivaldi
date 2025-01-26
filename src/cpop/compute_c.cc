#include <cassert>
#include <iostream>
#include "cuda_runtime.h"
#include "mpi.h"

#include "compute_c.hh"
#include "cusparse_helpers.hh"
#include "gpu_kernels.cuh"
#include "utils.hh"

namespace cpop {

int compute_c(cusparseHandle_t& handle, cusparseSpMatDescr_t& lV,
              cusparseDnMatDescr_t& ET, cusparseDnVecDescr_t* c_norm,
              MPI_Comm comm) {
  // Get MPI information
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Get input information
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  int64_t *csc_col_offset, *csc_row_inds;
  float *csc_values, *dn_values;
  cusparseIndexType_t offset_type, ind_type;
  cusparseIndexBase_t base_idx;
  cudaDataType sp_type, dn_type;
  cusparseOrder_t order;
  CHECK_CUSPARSE(cusparseCscGet(lV, &sp_rows, &sp_cols, &nnz,
                                (void**)&csc_col_offset, (void**)&csc_row_inds,
                                (void**)&csc_values, &offset_type, &ind_type,
                                &base_idx, &sp_type));
  CHECK_CUSPARSE(cusparseDnMatGet(ET, &dn_rows, &dn_cols, &ld,
                                  (void**)&dn_values, &dn_type, &order));

  assert(sp_rows == dn_rows && sp_cols == dn_cols);

  float* dz;
  CHECK_CUDA(cudaMalloc(&dz, dn_cols * sizeof(float)));
  // Because we're using CSC, the cluster assignments vector is exactly the
  // CSC row indices vector of V
  compute_z_vector(dn_cols, dz, csc_row_inds, dn_values);
  cusparseDnVecDescr_t z;  // local z
  CHECK_CUSPARSE(cusparseCreateDnVec(&z, dn_cols, dz, CUDA_R_32F));

  float alpha = 1.0f;
  float beta = 0.0f;

  float* dc_norm;
  CHECK_CUDA(cudaMalloc((void**)&dc_norm, sp_rows * sizeof(float)));
  CHECK_CUSPARSE(cusparseCreateDnVec(c_norm, sp_rows, dc_norm, CUDA_R_32F));

  // allocate an external buffer if needed
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lV, z, &beta, *c_norm,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // execute SpMV
  CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              lV, z, &beta, *c_norm, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

  // cleanup
  CHECK_CUDA(cudaFree(dz));
  CHECK_CUSPARSE(cusparseDestroyDnVec(z));
  CHECK_CUDA(cudaFree(dBuffer))
  return 0;
}

}  // namespace cpop
