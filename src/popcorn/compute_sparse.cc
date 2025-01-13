#include "compute_sparse.hh"

void popcorn::extract_kernel_tiles(float** tiles, slate::Matrix<float>& K,
                                   int col) {
  cudaMalloc(tiles, K.m() * K.tileNb(col) * sizeof(float));

  int64_t offset = 0;
  for (int64_t j = 0; j < K.mt(); ++j) {
    // Tiles guaranteed to be local
    slate::Tile<float> tile = K.at(j, col, K.tileDevice(j, col));
    cudaMemcpy(*tiles + offset, tile.data(),
               tile.mb() * tile.nb() * sizeof(float), cudaMemcpyDeviceToDevice);
    offset += tile.mb() * tile.nb();
  }
}

cusparseSpMatDescr_t popcorn::initialize_v(cusparseHandle_t& handle,
                                           int n, int k) {
  std::vector<int> rows, cols;
  std::vector<float> vals;

  for (int c = 0; c < n; ++c) {
    int r = c % k;
    int l = (n / k) + ((r < n % k) ? 1 : 0);
    rows.push_back(r);
    cols.push_back(c);
    vals.push_back(1.0f / l);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  void *cooRowInds, *cooColInds, *cooValues;

  cudaMalloc(&cooRowInds, n * sizeof(int));
  cudaMalloc(&cooColInds, n * sizeof(int));
  cudaMalloc(&cooValues, n * sizeof(float));

  cudaMemcpy(&cooRowInds, rows.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&cooColInds, cols.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&cooValues, vals.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  cusparseSpMatDescr_t V;
  cusparseCreateCoo(&V, k, n, n, cooRowInds, cooColInds, cooValues,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // TODO: Move these somewhere else (can't free until after use)
  // cudaFree(cooRowInd);
  // cudaFree(cooColInd);
  // cudaFree(cooValues);

  return V;
}

cusparseDnMatDescr_t popcorn::spmm(cusparseHandle_t& handle,
                                   cusparseSpMatDescr_t V,
                                   cusparseDnMatDescr_t K) {
  // Get input information
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  cudaDataType type;
  cusparseOrder_t order;
  float* values;
  cusparseSpMatGetSize(V, &sp_rows, &sp_cols, &nnz);
  cusparseDnMatGet(K, &dn_rows, &dn_cols, &ld, (void**) &values, &type, &order);

  assert(sp_cols == dn_rows && "Inner dimension must be equal in size.");
  assert(type == CUDA_R_32F && "Matrix data must be FP32.");

  // Define constants
  float alpha = 1.0f;
  float beta = 0.0f;

  // Allocate memory for output
  float* ET;
  cudaMalloc(&ET, sp_rows * dn_cols * sizeof(float));
  cusparseDnMatDescr_t ET_desc;
  cusparseCreateDnMat(&ET_desc, sp_rows, dn_cols, ld, ET, type, order);

  // Allocate workspace buffer
  size_t buffer_size;
  void* buffer;
  cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V,
                          K, &beta, ET_desc, CUDA_R_32F,
                          CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
  cudaMalloc(&buffer, buffer_size);

  // Perform SpMM
  cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, V, K, &beta,
               ET_desc, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  print_device_buffer(ET, sp_rows * dn_cols, 0);

  cudaFree(buffer);

  return ET_desc;
}