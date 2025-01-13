#include "compute_kernel.hh"

slate::Matrix<float> popcorn::load_matrix(const char* fname, int64_t rows,
                                          int64_t cols, int rank, int size) {
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Should be fine for reading even very large files, in order of millions for both rows and cols
  float* buf = (float*)malloc(cols * rows * sizeof(float));
  MPI_File_read_all(fh, buf, cols * rows, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create empty SLATE matrix object of size equal to data
  auto M = slate::Matrix<float>(cols, rows, cols, rows / size, 1, size,
                                MPI_COMM_WORLD);
  M.insertLocalTiles(slate::Target::Devices);

  // Fill data
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {
    for (int64_t i = 0; i < M.mt(); ++i) {
      if (M.tileIsLocal(i, j)) {
        slate::Tile<float> tile = M.at(i, j, M.tileDevice(i, j));
        int64_t stride = tile.stride();
        float* data = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {
          int64_t global_col = j * nb + jj;
          int64_t global_row = i * mb;
          cudaMemcpy(data + jj * stride, buf + global_col * m + global_row,
                     sizeof(float) * tile.mb(), cudaMemcpyHostToDevice);
        }
      }
    }
  }

  free(buf);
  MPI_File_close(&fh);

  return M;
}

slate::Matrix<float> popcorn::compute_kernel_matrix(slate::Matrix<float>& PT,
                                                    int rank, int size) {
  auto P = slate::transpose(PT);

  // Perform GEMM
  auto K = slate::Matrix<float>(P.m(), P.m(), P.tileMb(0), P.tileMb(0), 1, size,
                                MPI_COMM_WORLD);
  K.insertLocalTiles(slate::Target::Devices);
  slate::gemm<float>(1.0f, P, PT, 0.0f, K,
                     {{slate::Option::Target, slate::Target::Devices}});

  for (int64_t j = 0; j < K.nt(); ++j) {
    for (int64_t i = 0; i < K.mt(); ++i) {
      if (K.tileIsLocal(i, j)) {
        slate::Tile<float> tile = K.at(i, j, K.tileDevice(i, j));
        float* data = tile.data();
        launch_polynomial_kernel(tile.mb(), tile.nb(), data, 1.0f, 1.0f, 1.0f);
      }
    }
  }

  return K;
}