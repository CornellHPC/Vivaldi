#include "compute_c.hh"
#include "utils.hh"

cusparseDnVecDescr_t popcorn::compute_c(cusparseHandle_t& handle,
                                        cusparseSpMatDescr_t V,
                                        cusparseDnMatDescr_t ET,
                                        MPI_Comm comm) {
  // Get MPI information
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Get input information
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  cudaDataType type;
  cusparseOrder_t order;
  float* dn_values;
  cusparseSpMatGetSize(V, &sp_rows, &sp_cols, &nnz);
  cusparseDnMatGet(ET, &dn_rows, &dn_cols, &ld, (void**)&dn_values, &type,
                   &order);

  float* values;
  cudaMalloc(&values, dn_cols * sizeof(float));

  cusparseDnVecDescr_t c;
  cusparseCreateDnVec(&c, dn_cols, values, CUDA_R_32F);

  // Get nz row and col vectors and pass to mask kernel which:
  //   Gets submatrix of V
  //   Masks ET with submatrix of V
  //   Returns device pointer to z
  // Do local SpMV with V and z
  // MPI all-reduce sum (need to copy to host for now)

  return c;
}
