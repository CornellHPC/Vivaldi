#include "cpop_slate.hh"

#include "slate/slate.hh"

#include "../utils/utils.hh"

slate_matrix popcorn::load_data(const char* fname, int64_t rows, int64_t cols) {
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Should be fine for reading even very large files
  float* buf = (float*)malloc(cols * rows * sizeof(float));
  MPI_File_read_all(fh, buf, cols * rows, MPI_FLOAT, MPI_STATUS_IGNORE);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Create empty SLATE matrix object of size equal to data
  // Process grid: [ p1 | p2 | ... | pn ] to partition K the same way later
  auto PT = std::make_unique<slate::Matrix<float>>(
      cols, rows, cols, rows / size, 1, size, MPI_COMM_WORLD);
  PT->insertLocalTiles(slate::Target::Devices);

  // Fill data
  int64_t m = PT->m(), n = PT->n(), mb = PT->tileMb(0), nb = PT->tileNb(0);
  for (int64_t j = 0; j < PT->nt(); ++j) {
    for (int64_t i = 0; i < PT->mt(); ++i) {
      if (PT->tileIsLocal(i, j)) {
        slate::Tile<float> tile = PT->at(i, j, PT->tileDevice(i, j));
        int64_t stride = tile.stride();
        float* tile_data = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {
          int64_t global_col = j * nb + jj;
          int64_t global_row = i * mb;
          cudaMemcpy(tile_data + jj * stride, buf + global_col * m + global_row,
                     sizeof(float) * tile.mb(), cudaMemcpyHostToDevice);
        }
      }
    }
  }

  free(buf);
  MPI_File_close(&fh);

  return PT;
}

slate_matrix popcorn::compute_kernel_matrix(slate_matrix& PT, Kernel& kernel_func) {
  auto P = std::make_unique<slate::Matrix<float>>();
  *P = slate::transpose(*PT);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto comm = P->mpiComm();
  int grid_dim = square_grid_dim(comm);
  auto K = std::make_unique<slate::Matrix<DATA_TYPE>>(
      P->m(), P->m(), P->tileMb(0), P->tileMb(0), 1, size, comm);
  K->insertLocalTiles(slate::Target::Devices);
  slate::gemm<DATA_TYPE>(1.0f, *P, *PT, 0.0f, *K,
                         {{slate::Option::Target, slate::Target::Devices}});
  for (int64_t j = 0; j < K->nt(); ++j) {
    for (int64_t i = 0; i < K->mt(); ++i) {
      if (K->tileIsLocal(i, j)) {
        std::cout << i << " " << j << " " << rank << std::endl;
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