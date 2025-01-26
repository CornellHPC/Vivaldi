#include "cusparse_helpers.hh"

namespace cpop {

void print_arr_(float* arr, int64_t size) {
  std::cout << "[ ";
  for (int i = 0; i < size; ++i)
    std::cout << arr[i] << " ";
  std::cout << "]" << std::endl;
}

void print_arr_(int64_t* arr, int64_t size) {
  std::cout << "[ ";
  for (int i = 0; i < size; ++i)
    std::cout << arr[i] << " ";
  std::cout << "]" << std::endl;
}

void print_arr_(int* arr, int64_t size) {
  std::cout << "[ ";
  for (int i = 0; i < size; ++i)
    std::cout << arr[i] << " ";
  std::cout << "]" << std::endl;
}

void print(cusparseDnMatDescr_t& M, int rank) {
  // Get input information
  int64_t rows, cols, ld;
  void* values;
  cudaDataType type;
  cusparseOrder_t order;
  cusparseDnMatGet(M, &rows, &cols, &ld, &values, &type, &order);

  // Print resources
  std::cout << "Rank " << rank << ": Dense matrix has size " << rows << "x"
            << cols << std::endl;

  float* values_loc = (float*)malloc(rows * cols * sizeof(float));
  cudaMemcpy(values_loc, values, rows * cols * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "[ ";
  for (int i = 0; i < rows; ++i) {
    std::cout << "[ ";
    for (int j = 0; j < cols; ++j) {
      std::cout << values_loc[i * cols + j] << " ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "] " << std::endl;
  free(values_loc);
}

void print(cusparseDnVecDescr_t& M, int rank) {
  int64_t size;
  float* values;
  cudaDataType_t data_type;
  cusparseDnVecGet(M, &size, (void**)&values, &data_type);
  float* values_loc = (float*)malloc(size * sizeof(float));
  cudaMemcpy(values_loc, values, size * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Rank " << rank << " Dense vector has size " << size
            << " and values [ ";
  for (int i = 0; i < size; ++i)
    std::cout << values_loc[i] << " ";
  std::cout << "]" << std::endl;
  free(values_loc);
}

void print(cusparseSpMatDescr_t& M, int rank) {
  // Get input information
  int64_t rows, cols, nnz;
  int64_t *col_offsets, *row_inds;
  float* values;
  cusparseIndexType_t col_offsets_type, row_inds_type;
  cusparseIndexBase_t base_idx;
  cudaDataType_t data_type;
  cusparseCscGet(M, &rows, &cols, &nnz, (void**)&col_offsets, (void**)&row_inds,
                 (void**)&values, &col_offsets_type, &row_inds_type, &base_idx,
                 &data_type);

  // Print resources
  std::cout << "Rank " << rank << ": Sparse matrix has size " << rows << "x"
            << cols << " with " << nnz << " nnz" << std::endl;

  int64_t* col_offsets_loc = (int64_t*)malloc((cols + 1) * sizeof(int64_t));
  cudaMemcpy(col_offsets_loc, col_offsets, (cols + 1) * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  int64_t* row_inds_loc = (int64_t*)malloc(nnz * sizeof(int64_t));
  cudaMemcpy(row_inds_loc, row_inds, nnz * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  float* values_loc = (float*)malloc(nnz * sizeof(float));
  cudaMemcpy(values_loc, values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Col Offsets for rank " << rank << ": ";
  print_arr_(col_offsets_loc, cols + 1);
  std::cout << "Row Indices for rank " << rank << ": ";
  print_arr_(row_inds_loc, nnz);
  std::cout << "Values for rank " << rank << ": ";
  print_arr_(values_loc, nnz);

  free(col_offsets_loc);
  free(row_inds_loc);
  free(values_loc);
}

void print(cusparseSpMatDescr_t& M) {
  // Get input information
  int64_t rows, cols, nnz;
  int64_t *row_offsets, *col_inds;
  float* values;
  cusparseIndexType_t row_offsets_type, col_inds_type;
  cusparseIndexBase_t base_idx;
  cudaDataType_t data_type;
  cusparseCsrGet(M, &rows, &cols, &nnz, (void**)&row_offsets, (void**)&col_inds,
                 (void**)&values, &row_offsets_type, &col_inds_type, &base_idx,
                 &data_type);

  // Print resources
  std::cout << "Global sparse matrix has size " << rows << "x" << cols
            << " with " << nnz << " nnz" << std::endl;

  int64_t* row_offsets_loc = (int64_t*)malloc((rows + 1) * sizeof(int64_t));
  cudaMemcpy(row_offsets_loc, row_offsets, (rows + 1) * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  int64_t* col_inds_loc = (int64_t*)malloc(nnz * sizeof(int64_t));
  cudaMemcpy(col_inds_loc, col_inds, nnz * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  float* values_loc = (float*)malloc(nnz * sizeof(float));
  cudaMemcpy(values_loc, values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Global Row Offsets: ";
  print_arr_(row_offsets_loc, rows + 1);
  std::cout << "Global Column Indices: ";
  print_arr_(col_inds_loc, nnz);
  std::cout << "Global Values: ";
  print_arr_(values_loc, nnz);

  free(row_offsets_loc);
  free(col_inds_loc);
  free(values_loc);
}

}  // namespace cpop