#include "compute_sparse.hh"

void popcorn::extract_kernel_tiles(float* tiles, slate_matrix& K, int col) {
  cudaMalloc(&tiles, K->m() * K->tileNb(col) * sizeof(float));

  int64_t offset = 0;
  for (int64_t j = 0; j < K->mt(); ++j) {
    // Tiles guaranteed to be local
    slate::Tile<float> tile = K->at(j, col, K->tileDevice(j, col));
    cudaMemcpy(tiles + offset, tile.data(),
               tile.mb() * tile.nb() * sizeof(float), cudaMemcpyDeviceToDevice);
    offset += tile.mb() * tile.nb();
  }
}

cusparseSpMatDescr_t popcorn::initialize_v(cusparseHandle_t& cusparse_handle,
                                           int m, int k) {
  std::vector<int> rows, cols;
  std::vector<float> val;

  // TODO: Speed up initialization
  for (int c = 0; c < m; ++c) {
    int r = c % k;
    int l = (m / k) + ((r < m % k) ? 1 : 0);
    rows.push_back(r);
    cols.push_back(c);
    val.push_back(1.0f / l);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // if (rank == 0) {
  //   for (int i = 0; i < m; ++i) {
  //     std::cout << "(" << row.at(i) << "," << col.at(i) << "," << val.at(i)
  //               << ")" << std::endl;
  //   }
  // }

  void *cooRowInd, *cooColInd, *cooValues;

  cudaMalloc(&cooRowInd, m * sizeof(int));
  cudaMalloc(&cooColInd, m * sizeof(int));
  cudaMalloc(&cooValues, m * sizeof(float));

  cusparseSpMatDescr_t V;
  cusparseCreateCoo(&V, k, m, m, cooRowInd, cooColInd, cooValues,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // TODO: Move these somewhere else (can't free until after use)
  // cudaFree(cooRowInd);
  // cudaFree(cooColInd);
  // cudaFree(cooValues);

  return V;
}

cusparseDnMatDescr_t popcorn::spmm(cusparseHandle_t& cusparse_handle,
                                   cusparseSpMatDescr_t sparse,
                                   cusparseDnMatDescr_t dense) {
  // Get input information
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  cudaDataType type;
  cusparseOrder_t order;
  cusparseSpMatGetSize(sparse, &sp_rows, &sp_cols, &nnz);
  cusparseDnMatGet(dense, &dn_rows, &dn_cols, &ld, nullptr, &type, &order);

  // Protect against bad input
  assert(sp_cols == dn_rows && "Inner dimension must be equal in size.");
  assert(type == CUDA_R_32F && "Matrix data must be FP32.");

  // Define constants
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate memory for output
  float* out_data;
  cudaMalloc(&out_data, sp_rows * dn_cols * sizeof(float));
  cusparseDnMatDescr_t out;
  cusparseCreateDnMat(&out, sp_rows, dn_cols, ld, out_data, type, order);

  // Allocate workspace
  size_t buffer_size;
  void* buffer;
  cusparseSpMM_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse,
                          dense, &beta, out, CUDA_R_32F,
                          CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
  cudaMalloc(&buffer, buffer_size);

  // Perform SpMM
  cusparseSpMM(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse, dense, &beta,
               out, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  // Release workspace
  cudaFree(buffer);

  return out;
}