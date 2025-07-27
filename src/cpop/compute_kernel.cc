#include "mpi.h"

#include "compute_kernel.hh"
#include "gpu_kernels.cuh"
#include "utils.hh"

namespace cpop {

slate::Matrix<float> load_matrix(const char* fname, int64_t rows, int64_t cols,
                                 MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_File fh;
  MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Should be fine for reading even very large files, in order of millions for both rows and cols
  float* buf = (float*)malloc(cols * rows * sizeof(float));
  MPI_File_read_all(fh, buf, cols * rows, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create empty SLATE matrix object of size equal to data
  int t = rows / size + (rows > size * size && rows % size > 0);
  auto M = slate::Matrix<float>(cols, rows, cols, t, 1, size, comm);
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

float* compute_kernel_matrix(slate::Matrix<float>& PT, float gamma, float c,
                             float r) {
  int rank, size;
  MPI_Comm_rank(PT.mpiComm(), &rank);
  MPI_Comm_size(PT.mpiComm(), &size);

  assert(PT.n() > size * size && "TODO: Handle case with remainder.");

  // Create local K buffer
  float* data;
  int64_t rows = PT.n();
  int* t_sizes = compute_tile_sizes(rows, size);
  CHECK_CUDA(cudaMalloc(&data, rows * t_sizes[rank] * sizeof(float)));

  // Initialize matrices
  auto P = slate::transpose(PT);
  auto K = slate::Matrix<float>(P.m(), P.m(), P.tileMb(0), P.tileMb(0), 1, size,
                                PT.mpiComm());

  // Fill K matrix with tiles using buffer
  for (int64_t i = 0; i < size; ++i) {
    int offset = P.tileMb(0) * t_sizes[rank] * i;
    K.tileInsert(i, rank, 0, data + offset, t_sizes[i]);
  }

  // Compute kernel matrix
  slate::gemm<float>(1.0f, P, PT, 0.0f, K,
                     {{slate::Option::Target, slate::Target::Devices}});
  K.tileLayoutConvertOnDevices(blas::Layout::RowMajor);
  launch_polynomial_kernel(rows, t_sizes[rank], data, gamma, c, r);

  // Clean up and return
  free(t_sizes);
  return data;
}

}  // namespace cpop
