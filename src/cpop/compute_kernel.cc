#include <fstream>
#include "mpi.h"

#include "compute_kernel.hh"
#include "gpu_kernels.cuh"

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
  int p = std::floor(std::sqrt(size));
  int q = std::floor(std::sqrt(size));
  if (p * q != size) {
    std::cout << "WARNING: The number of ranks is not a perfect square."
              << std::endl;
  }
  int rows_per_block = (rows / size) + ((rows % size == 0) ? 0 : 1);
  int cols_per_block = (cols / size) + ((cols % size == 0) ? 0 : 1);
  auto M = slate::Matrix<float>(cols, rows, cols_per_block, rows_per_block, p,
                                q, comm);
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

  // Perform GEMM
  auto P = slate::transpose(PT);

  // std::cout << "Multiplying a " << P.m() << "x" << P.n() << " matrix with a "
  //           << PT.n() << "x" << PT.m() << " matrix on rank " << rank << " with "
  //           << "tiles of size " << P.tileMb(0) << "x" << P.tileNb(0) << " and "
  //           << PT.tileMb(0) << "x" << PT.tileNb(0) << std::endl;

  int grid_size = std::floor(std::sqrt(size));
  int tile_size = P.tileMb(0);
  auto K = slate::Matrix<float>(P.m(), P.m(), tile_size, tile_size,
                                slate::GridOrder::Row, grid_size, grid_size,
                                PT.mpiComm());
  K.insertLocalTiles(slate::Target::Devices);
  slate::gemm<float>(1.0f, P, PT, 0.0f, K,
                     {{slate::Option::Target, slate::Target::Devices}});

  // Ensure tiles are row-major
  K.tileLayoutConvertOnDevices(blas::Layout::RowMajor);

  // std::cout << "Gemm Completed" << std::endl;

  float* horizontal_tile = nullptr;
  // std::cout << "Allocating memory for horizontal tile on rank " << rank
  //           << " with size " << K.m() * tile_size * sizeof(float) << std::endl;
  cudaMalloc(&horizontal_tile, K.m() * tile_size * sizeof(float));
  if (!horizontal_tile) {
    std::cerr << "Failed to allocate memory for horizontal tile on rank "
              << rank << std::endl;
    return nullptr;
  }

  for (int64_t j = 0; j < K.nt(); ++j) {
    for (int64_t i = 0; i < K.mt(); ++i) {
      if (K.tileIsLocal(i, j)) {
        // std::cout << "On rank " << rank << ", processing tile (" << i << ", "
        //           << j << ") " << std::endl;
        slate::Tile<float> tile = K.at(i, j, K.tileDevice(i, j));
        float* data = tile.data();
        if (j != rank) {
          // std::cout << "Tile (" << i << ", " << j
          //           << ") is not in the local column for rank " << rank
          //           << " and needs to be moved to rank " << j << std::endl;
          MPI_Send(data, tile_size * tile_size, MPI_FLOAT, j, i, K.mpiComm());
        } else {
          int offset = tile_size * tile_size * i;
          cudaMemcpy(horizontal_tile + offset, data,
                     tile_size * tile_size * sizeof(float),
                     cudaMemcpyDeviceToDevice);
          // std::cout << "Tile (" << i << ", " << j
          //           << ") is in the local column for rank " << rank
          //           << ", copied to horizontal tile at offset " << offset
          //           << std::endl;
        }

        // slate::Tile<float> tile = K.at(i, j, K.tileDevice(i, j));
        // float* data = tile.data();

        // launch_polynomial_kernel(tile.mb(), tile.nb(), data, gamma, c, r);
      } else if (j == rank) {
        // std::cout << "Tile (" << i << ", " << j
        //           << ") is in the local column for rank " << rank
        //           << ", need to receive." << std::endl;

        int offset = tile_size * tile_size * i;
        int source_rank = K.tileRank(i, j);
        // std::cout << "Receiving tile (" << i << ", " << j << ") from rank "
        //           << source_rank << " on rank " << rank << " at offset "
        //           << offset << std::endl;
        MPI_Recv(horizontal_tile + offset, tile_size * tile_size, MPI_FLOAT,
                 source_rank, i, K.mpiComm(), MPI_STATUS_IGNORE);
        // std::cout << "Tile (" << i << ", " << j << ") received on rank " << rank
        //           << std::endl;
      }
    }
  }

  // std::cout << "Kernel Resorting Completed" << std::endl;
  launch_polynomial_kernel(K.m(), tile_size, horizontal_tile, gamma, c, r);
  // std::cout << "Kernel Computation Completed" << std::endl;

  // std::cout << "Horizontal tile on rank " << rank << " is ready with size "
  //           << K.m() << " by " << tile_size << std::endl;

  // float* values = horizontal_tile;
  // int64_t count = K.m() * tile_size;

  // std::ofstream out_file("2d_gemm_" + std::to_string(rank) + ".txt");
  // std::cout << "Opening file: 2d_gemm_" << rank << ".txt" << std::endl;
  // if (out_file.is_open()) {
  //   std::cout << "Writing values to file." << std::endl;
  //   float* host_values = (float*)malloc(count * sizeof(float));
  //   cudaMemcpy(host_values, values, count * sizeof(float),
  //              cudaMemcpyDeviceToHost);
  //   for (int i = 0; i < count; ++i) {
  //     out_file << std::fixed << std::setprecision(2) << host_values[i] << "\n";
  //   }
  //   out_file.close();
  // } else {
  //   std::cerr << "Could not open file for writing buffer mismatch."
  //             << std::endl;
  // }

  // Return early (there will be no remainder tiles)
  return horizontal_tile;
}

}  // namespace cpop
