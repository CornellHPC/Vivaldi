#include "compute_sparse.hh"

void popcorn::create_kernel_descriptors(cusparseDnMatDescr_t K1_desc,
                                  cusparseDnMatDescr_t K2_desc, float* K1,
                                  float* K2, slate_matrix& K, bool* multiple) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cudaMalloc(&K1, K->n() * K->tileMb(rank) * sizeof(float));
  if (rank + size < K->mt())
    cudaMalloc(&K2, K->n() * K->tileMb(rank + size) * sizeof(float));

  int64_t offset = 0;
  for (int64_t j = 0; j < K->nt(); ++j) {
    // Tiles guaranteed to be local
    slate::Tile<float> tile = K->at(rank, j, K->tileDevice(rank, j));
    cudaMemcpy(K1 + offset, tile.data(), tile.mb() * tile.nb() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    offset += tile.mb() * tile.nb();
  }

  cusparseCreateDnMat(&K1_desc, K->tileMb(rank), K->n(), K->tileMb(rank), K1,
                      CUDA_R_32F, CUSPARSE_ORDER_COL);

  if (rank + size < K->mt()) {
    *multiple = true;
    offset = 0;
    for (int64_t j = 0; j < K->nt(); ++j) {
      // Tiles guaranteed to be local
      slate::Tile<float> tile =
          K->at(rank + size, j, K->tileDevice(rank + size, j));
      cudaMemcpy(K2 + offset, tile.data(),
                 tile.mb() * tile.nb() * sizeof(float),
                 cudaMemcpyDeviceToDevice);
      offset += tile.mb() * tile.nb();
    }

    cusparseCreateDnMat(&K2_desc, K->tileMb(rank + size), K->n(),
                      K->tileMb(rank + size), K2, CUDA_R_32F,
                      CUSPARSE_ORDER_COL);
  }
}