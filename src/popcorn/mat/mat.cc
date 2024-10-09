#include "../utils/utils.hh"
#include "dense_mat.hh"
#include "sparse_mat.hh"

using namespace popcorn;

void fill_slate_mat_with_buffer(slate::Matrix<DATA_TYPE>* M, DATA_TYPE* buf) {
  int64_t m = M->m(), n = M->n(), mb = M->tileMb(0), nb = M->tileNb(0);
  for (int64_t j = 0; j < M->nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < M->mt(); ++i) {  // j loops over block rows
      if (M->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = M->at(i, j, M->tileDevice(i, j));
        int64_t lda = tile.stride();
        DATA_TYPE* A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {  // jj loops over columns
          int64_t global_column = j * nb + jj;
          int64_t global_row_start = i * mb;
          cudaMemcpy(A + jj * lda, buf + global_column * m + global_row_start,
                     sizeof(DATA_TYPE) * tile.mb(), cudaMemcpyHostToDevice);
        }
      }
    }
  }
}

DenseMat* DenseMat::load_from_file(const char* filename, int64_t rows,
                                   int64_t cols, int64_t rows_per_block,
                                   int64_t cols_per_block, int64_t grid_dim,
                                   MPI_Comm comm) {
  // Open file
  MPI_File fh;
  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = rows * cols * sizeof(DATA_TYPE);
  DATA_TYPE* data = (DATA_TYPE*)malloc(count);
  MPI_File_read_all(fh, data, rows * cols, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create matrix
  auto M = new slate::Matrix<DATA_TYPE>(
      rows, cols, rows_per_block, cols_per_block, grid_dim, grid_dim, comm);
  M->insertLocalTiles(slate::Target::Devices);

  // Initialize data
  fill_slate_mat_with_buffer(M, data);

  // Clean up
  free(data);
  MPI_File_close(&fh);

  return DenseMat::from_slate(M, rows_per_block, cols_per_block, grid_dim);
}

DenseMat* DenseMat::from_slate(slate::Matrix<DATA_TYPE>* sm,
                               int64_t rows_per_block, int64_t cols_per_block,
                               int64_t grid_dim) {
  return new DenseMat(sm->m(), sm->n(), sm->mt(), sm->nt(), rows_per_block,
                      cols_per_block, grid_dim, sm->mpiComm(), sm, NULL);
}