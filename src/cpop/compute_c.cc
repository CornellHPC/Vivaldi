#include <cassert>
#include <iostream>
#include "cuda_runtime.h"
#include "mpi.h"

#include "compute_c.hh"
#include "cusparse_helpers.hh"
#include "gpu_kernels.cuh"
#include "utils.hh"

namespace cpop {

int init_c(int k, cusparseDnVecDescr_t* c) {
  float* dc;
  CHECK_CUDA(cudaMalloc((void**)&dc, k * sizeof(float)));
  CHECK_CUSPARSE(cusparseCreateDnVec(c, k, dc, CUDA_R_32F));
  return EXIT_SUCCESS;
}

int init_z(int t, cusparseDnVecDescr_t* z) {
  float* dz;
  CHECK_CUDA(cudaMalloc(&dz, t * sizeof(float)));
  CHECK_CUSPARSE(cusparseCreateDnVec(z, t, dz, CUDA_R_32F));
  return EXIT_SUCCESS;
}

int compute_z(cusparseSpMatDescr_t& lV, cusparseDnMatDescr_t& ET,
              cusparseDnVecDescr_t& z) {
  // Get input information
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  int64_t *csc_col_offset, *csc_row_inds;
  float *csc_values, *dn_values, *dz;
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
  CHECK_CUSPARSE(cusparseDnVecGetValues(z, (void**)&dz));

  // Because we're using CSC, the cluster assignments vector is exactly the
  // CSC row indices vector of V
  launch_z_kernel(dn_cols, dz, csc_row_inds, dn_values);
  return EXIT_SUCCESS;
}

int spmv(cusparseHandle_t& handle, cusparseSpMatDescr_t& lV,
         cusparseDnVecDescr_t& z, cusparseDnVecDescr_t& c) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // allocate an external buffer if needed
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, lV, z, &beta, c,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // execute SpMV
  CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              lV, z, &beta, c, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

  // cleanup
  CHECK_CUDA(cudaFree(dBuffer));

  return EXIT_SUCCESS;
}

}  // namespace cpop
