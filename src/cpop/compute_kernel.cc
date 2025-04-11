#include "mpi.h"

#include "compute_kernel.hh"
#include "gpu_kernels.cuh"

namespace cpop {

slate::Matrix<float> load_matrix(const char* fname, int64_t rows, int64_t cols,
                                 MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // int grid_size = std::floor(std::sqrt(size));
  // int64_t cols_per_block =
  //     (cols / grid_size) + ((cols % grid_size == 0) ? 0 : 1);
  // int64_t rows_per_block =
  //     (rows / grid_size) + ((rows % grid_size == 0) ? 0 : 1);

  MPI_File fh;
  MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Should be fine for reading even very large files, in order of millions for both rows and cols
  float* buf = (float*)malloc(cols * rows * sizeof(float));
  MPI_File_read_all(fh, buf, cols * rows, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create empty SLATE matrix object of size equal to data
  // int t = rows / size + (rows > size * size && rows % size > 0);
  // auto M = slate::Matrix<float>(cols, rows, cols, t, 1, size, comm);

  int p = std::floor(std::sqrt(size));
  int q = std::floor(std::sqrt(size));
  if (p * q != size) {
    std::cout << "WARNING: The number of ranks is not a perfect square."
              << std::endl;
  }
  auto M = slate::Matrix<float>(cols, rows, );

  // auto M = slate::Matrix<float>(cols, rows, cols_per_block, rows_per_block,
  //                               grid_size, grid_size, comm);
  std::cout << "Initializing P with " << cols << " " << rows << " "
            << cols_per_block << " " << rows_per_block << " " << grid_size
            << std::endl;
  std::cout << "Tile:" << M.tileMb(0) << " " << M.tileNb(0) << std::endl;
  std::cout << "Size:" << M.mt() << " " << M.nt() << std::endl;
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

int64_t extract_kernel_tiles(float** tiles, slate::Matrix<float>& K, int col) {
  int64_t elems = K.m() * K.tileNb(col);
  cudaMalloc(tiles, elems * sizeof(float));

  int64_t offset = 0;
  for (int64_t j = 0; j < K.mt(); ++j) {
    // Tiles guaranteed to be local
    slate::Tile<float> tile = K.at(j, col, K.tileDevice(j, col));
    cudaMemcpy(*tiles + offset, tile.data(),
               tile.mb() * tile.nb() * sizeof(float), cudaMemcpyDeviceToDevice);
    offset += tile.mb() * tile.nb();
  }

  return elems;
}

float* compute_kernel_matrix(slate::Matrix<float>& PT, float gamma, float c,
                             float r) {
  int rank, size;
  MPI_Comm_rank(PT.mpiComm(), &rank);
  MPI_Comm_size(PT.mpiComm(), &size);

  // Perform GEMM
  auto P = slate::transpose(PT);
  int p_rows = P.m();
  int t = p_rows / size + (p_rows > size * size && p_rows % size > 0);
  std::cout << "Rank " << rank << " Printing P size: " << P.tileMb(0) << " "
            << P.tileNb(0) << " - t is still " << t << std::endl;
  int grid_size = std::floor(std::sqrt(size));
  auto K = slate::Matrix<float>(P.m(), P.m(), P.tileMb(0), P.tileMb(0),
                                grid_size, grid_size, PT.mpiComm());
  K.insertLocalTiles(slate::Target::Devices);
  slate::gemm<float>(1.0f, P, PT, 0.0f, K,
                     {{slate::Option::Target, slate::Target::Devices}});

  std::cout << "GEMM COMPLTED" << std::endl;

  for (int64_t j = 0; j < K.nt(); ++j) {
    for (int64_t i = 0; i < K.mt(); ++i) {
      std::cout << "Fetching tile " << i << " " << j << std::endl;
      if (K.tileIsLocal(i, j)) {
        slate::Tile<float> tile = K.at(i, j, K.tileDevice(i, j));
        float* data = tile.data();
        launch_polynomial_kernel(tile.mb(), tile.nb(), data, gamma, c, r);
      }
    }
  }

  std::cout << "POLYNOMIAL KERNEL COMPLTED" << std::endl;

  int count = 0;
  float* values;
  if (rank == 1) {
    // Ensure tiles are row-major
    K.tileLayoutConvertOnDevices(blas::Layout::RowMajor);

    // Copy tiles in local column to buffer
    int64_t count = P.tileMb(0) * P.tileNb(0);
    float* values = (float*)malloc(count * sizeof(float));
    // cudaMalloc(&values, count * sizeof(float));
    for (int64_t j = 0; j < K.nt(); ++j) {
      for (int64_t i = 0; i < K.mt(); ++i) {
        if (K.tileIsLocal(i, j)) {
          std::cout << "Copying tile " << i << " " << j << std::endl;
          slate::Tile<float> tile = K.at(i, j, K.tileDevice(i, j));
          float* data = tile.data();
          cudaMemcpy(values, data, count * sizeof(float),
                     cudaMemcpyDeviceToHost);
          std::cout << "Rank" << rank << " Values content:" << std::endl;
          for (int64_t i = 0; i < count; ++i) {
            std::cout << values[i] << " ";
          }
          std::cout << std::endl;
        }
      }
    }
    free(values);
  }

  std::cout << "Rank" << rank << " Finished ROWMAJOR" << std::endl;

  if (rank == 1) {
    // Ensure tiles are row-major
    K.tileLayoutConvertOnDevices(blas::Layout::ColMajor);

    // Copy tiles in local column to buffer
    int count = P.tileMb(0) * P.tileNb(0);
    float* values = (float*)malloc(count * sizeof(float));
    // cudaMalloc(&values, count * sizeof(float));
    for (int64_t j = 0; j < K.nt(); ++j) {
      for (int64_t i = 0; i < K.mt(); ++i) {
        if (K.tileIsLocal(i, j)) {
          std::cout << "Copying tile " << i << " " << j << std::endl;
          slate::Tile<float> tile = K.at(i, j, K.tileDevice(i, j));
          float* data = tile.data();
          cudaMemcpy(values, data, count * sizeof(float),
                     cudaMemcpyDeviceToHost);
          std::cout << "Rank" << rank << " Values content:" << std::endl;
          for (int64_t i = 0; i < count; ++i) {
            std::cout << values[i] << " ";
          }
          std::cout << std::endl;
        }
      }
    }
    free(values);
  }

  MPI_Abort(PT.mpiComm(), EXIT_FAILURE);

  // Return early if there is no remainder
  if (K.tileNb(0) * size >= K.n())
    return values;

  // Warn on remainder
  std::cout << "WARNING: The tiles could not be distributed without remainder."
            << std::endl;

  // Copy tiles in remainder column to buffer
  float* _remainder;
  int remainder_count = 0;
  if (rank + size < K.nt()) {
    float* remainder;
    remainder_count = (int)extract_kernel_tiles(&remainder, K, rank + size);
    _remainder = (float*)malloc(remainder_count * sizeof(float));
    cudaMemcpy(_remainder, remainder, remainder_count * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(remainder);
  }

  // Gather remainder buffer sizes on root (i.e. last rank)
  int root = size - 1;
  int* recvcounts;
  if (rank == root) {
    recvcounts = (int*)malloc(size * sizeof(int));
  }
  MPI_Gather(&remainder_count, 1, MPI_INT, recvcounts, 1, MPI_INT, root,
             PT.mpiComm());

  // Gather remainder buffers on root (i.e. last rank)
  int* displs;
  int64_t total = 0;
  float* recvbuf;
  if (rank == root) {
    displs = (int*)malloc(size * sizeof(int));
    displs[0] = 0;
    total = recvcounts[0];
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
      total += recvcounts[i];
    }
    recvbuf = (float*)malloc(total * sizeof(float));
  }
  MPI_Gatherv(_remainder, remainder_count, MPI_FLOAT, recvbuf, recvcounts,
              displs, MPI_FLOAT, root, PT.mpiComm());

  // Reconstruct root buffer (i.e. last rank)
  int64_t rows = K.m();
  int64_t cols = (count + total) / rows;
  if (rank == root) {
    // Copy local column to host
    float* _values = (float*)malloc(count * sizeof(float));
    cudaMemcpy(_values, values, count * sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate output
    float* tmp = (float*)malloc((count + total) * sizeof(float));

    // Fill output buffer
    int64_t offset = 0;
    for (int64_t i = 0; i < rows; ++i) {
      // Copy local row
      int64_t local_cols = K.tileNb(rank);
      for (int64_t j = 0; j < local_cols; ++j) {
        tmp[offset++] = _values[i * local_cols + j];
      }

      // Copy remainder rows
      for (int64_t p = 0; p < size; ++p) {
        int64_t remainder_cols = recvcounts[p] / rows;
        for (int64_t j = 0; j < remainder_cols; ++j) {
          tmp[offset++] = recvbuf[displs[p] + i * remainder_cols + j];
        }
      }
    }

    // Move data to device
    cudaFree(values);
    cudaMalloc(&values, (count + total) * sizeof(float));
    cudaMemcpy(values, tmp, (count + total) * sizeof(float),
               cudaMemcpyHostToDevice);

    // Free host resources
    free(recvcounts);
    free(displs);
    free(recvbuf);
    free(_values);
    free(tmp);
  }
  if (rank + size < K.nt()) {
    free(_remainder);
  }

  // Create cuSPARSE dense matrix descriptors
  return values;
}

}  // namespace cpop
