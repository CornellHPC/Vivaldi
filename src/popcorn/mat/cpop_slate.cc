#include "cpop_slate.hh"

// Local imports
#include "../utils/utils.hh"

sm_ptr popcorn::load_from_file(const char* filename, int64_t rows, int64_t cols,
                               MPI_Comm comm) {
  // Open file
  MPI_File fh;
  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = cols * rows * sizeof(DATA_TYPE);
  DATA_TYPE* data = (DATA_TYPE*)malloc(count);
  MPI_File_read_all(fh, data, cols * rows, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Compute tile size
  int grid_dim = square_grid_dim(comm);
  int cols_per_block = tile_dim(comm, cols);
  int rows_per_block = tile_dim(comm, rows);

  // Create empty SLATE matrix object of size equal to data
  auto M = std::make_unique<slate::Matrix<DATA_TYPE>>(
      cols, rows, cols_per_block, rows_per_block, grid_dim, grid_dim, comm);
  M->insertLocalTiles(slate::Target::Devices);

  // Fill data
  int64_t m = M->m(), n = M->n(), mb = M->tileMb(0), nb = M->tileNb(0);
  for (int64_t j = 0; j < M->nt(); ++j) {    // j loops over block columns
    for (int64_t i = 0; i < M->mt(); ++i) {  // i loops over block rows
      if (M->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = M->at(i, j, M->tileDevice(i, j));
        int64_t lda = tile.stride();
        DATA_TYPE* A = tile.data();

        for (int64_t jj = 0; jj < tile.nb();
             ++jj) {  // jj loops over tile columns
          int64_t global_column = j * nb + jj;
          int64_t global_row_start = i * mb;
          cudaMemcpy(A + jj * lda, data + global_column * m + global_row_start,
                     sizeof(DATA_TYPE) * tile.mb(), cudaMemcpyHostToDevice);
        }
      }
    }
  }

  // Clean up
  free(data);
  MPI_File_close(&fh);

  return M;
}

sm_ptr popcorn::compute_k(sm_ptr& PT, Kernel& kernel_func) {
  auto P = std::make_unique<slate::Matrix<DATA_TYPE>>();
  *P = slate::transpose(*PT);

  auto comm = P->mpiComm();
  int grid_dim = square_grid_dim(comm);
  auto K = std::make_unique<slate::Matrix<DATA_TYPE>>(
      P->m(), P->m(), P->tileMb(0), P->tileMb(0), grid_dim, grid_dim, comm);
  K->insertLocalTiles(slate::Target::Devices);
  slate::gemm<DATA_TYPE>(1.0f, *P, *PT, 0.0f, *K,
                         {{slate::Option::Target, slate::Target::Devices}});
  for (int64_t j = 0; j < K->nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < K->mt(); ++i) {  // j loops over block rows
      if (K->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = K->at(i, j, K->tileDevice(i, j));
        DATA_TYPE* tile_buf = tile.data();
        kernel_func.f(tile.mb(), tile.nb(), tile_buf);
      }
    }
  }
  return K;
}

c_dn_ptr popcorn::to_combblas(sm_ptr& K) {
  // Initialize CombBLAS communicator grid
  MPI_Comm comm = K->mpiComm();
  int p = square_grid_dim(comm);
  auto grid = std::make_shared<combblas::CommGrid>(comm, p, p);

  // Initialize CombBLAS distributed dense matrix
  auto C = std::make_unique<combblas::DnParMat<int64_t, DATA_TYPE>>(
      grid, K->m(), K->n(), (DATA_TYPE)0);

  // Copy the tiles 1-to-1
  for (int64_t j = 0; j < K->nt(); ++j) {
    for (int64_t i = 0; i < K->mt(); ++i) {
      if (K->tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = K->at(i, j, K->tileDevice(i, j));
        C->getarr().resize(tile.size());

        DATA_TYPE* src = tile.data();
        DATA_TYPE* dst = C->getarr().data();
        cudaMemcpy(dst, src, sizeof(DATA_TYPE) * tile.size(),
                   cudaMemcpyDeviceToHost);
      }
    }
  }

  return C;
}
