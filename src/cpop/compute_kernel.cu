#include "mpi.h"

#include "compute_kernel.hh"
#include "gpu_kernels.cuh"
#include "utils.hh"

#include <unistd.h>

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

slate::Matrix<float> load_matrix2d(const char* fname, int64_t rows, int64_t cols,
                                     MPI_Comm comm) 
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_File fh;
  MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Should be fine for reading even very large files, in order of millions for both rows and cols
  float* buf = (float*)malloc(cols * rows * sizeof(float));
  MPI_File_read_all(fh, buf, cols * rows, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create empty SLATE matrix object of size equal to data
  int grid_dim = square_grid_dim(comm);
  int cols_per_block = tile_dim(comm, cols);
  int rows_per_block = tile_dim(comm, rows);
  auto M = slate::Matrix<float>(cols, rows, cols_per_block, rows_per_block, grid_dim, grid_dim, comm);
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

  //assert(PT.n() > size * size && "TODO: Handle case with remainder.");

  // Create local K buffer
  float* data;
  int64_t rows = PT.n();
  int* t_sizes = compute_tile_sizes(rows, size);
  int64_t my_size = t_sizes[rank];
  CHECK_CUDA(cudaMalloc(&data, rows * my_size * sizeof(float)));

  // Initialize matrices
  auto P = slate::transpose(PT);
  auto K = slate::Matrix<float>(P.m(), P.m(), P.tileMb(0), P.tileMb(0), 1, size,
                                PT.mpiComm());

  // Fill K matrix with tiles using buffer
  for (int64_t i = 0; i < size; ++i) {
    int64_t offset = ((int64_t)P.tileMb(0)) * my_size * i;
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

float* compute_kernel_matrix2d(Handle& handle, slate::Matrix<float>& PT, float gamma, float c, float r, bool redist) 
{
  int rank, size;
  MPI_Comm_rank(PT.mpiComm(), &rank);
  MPI_Comm_size(PT.mpiComm(), &size);

  int grid_dim = square_grid_dim(PT.mpiComm());
  int row_rank = rank / grid_dim;
  int col_rank = rank % grid_dim;


  // Create local K buffer
  float* data;
  int64_t rows = PT.n();
  auto t_rows = compute_tile_sizes(rows, grid_dim);
  auto t_cols = compute_tile_sizes(rows, grid_dim);
  int64_t loc_rows = t_rows[row_rank];
  int64_t loc_cols = t_cols[col_rank];
  CHECK_CUDA(cudaMalloc(&data,  loc_rows * loc_cols * sizeof(float)));

  int q = tile_dim(MPI_COMM_WORLD, rows);

  // Initialize matrices
  auto P = slate::transpose(PT);
  auto K = slate::Matrix<float>(P.m(), P.m(), P.tileMb(0), P.tileMb(0), 
                                grid_dim, grid_dim,
                                PT.mpiComm());

  // Fill K matrix with tiles using buffer
  K.tileInsert(col_rank, row_rank, 0, data, loc_rows);

  // Compute kernel matrix
  slate::gemm<float>(1.0f, P, PT, 0.0f, K,
                     {{slate::Option::Target, slate::Target::Devices}});
  if (!redist)
  {
      K.tileLayoutConvertOnDevices(blas::Layout::RowMajor);
  }
  else
  {
      K.tileLayoutConvertOnDevices(blas::Layout::ColMajor);
  }

  launch_polynomial_kernel(loc_rows, loc_cols, data, gamma, c, r);



#ifndef BASIC
  auto r_start = hrc::now();
#endif

  if (redist)
  {
      assert( (P.m() % size)==0 && "Redistribution only works without remainder for now");
      data = redistribute_2d_1d(handle, data, P.m(), P.tileMb(0));
  }

#ifndef BASIC
  timer.k_redist = get_time_elapsed(r_start);
#endif


  // Clean up and return
  return data;
}


float * redistribute_2d_1d(Handle& handle, float * K, const uint64_t m, const uint64_t mb)
{

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int grid_dim = square_grid_dim(MPI_COMM_WORLD);
    int col_rank = rank / grid_dim;

    int * sendcounts = new int[size];
    memset(sendcounts, 0, sizeof(int) * size);
    int * senddispls = new int[size];
    memset(senddispls, 0, sizeof(int) * size);

    int64_t slice_size = mb * (m/size); 

    // Iterate through my column major blocks and figure out how many are to be sent to other processes
    // I have sqrt(P) col blocks
    for (int i=0; i<grid_dim; i++)
    {
        int owner = i + col_rank*grid_dim;
        sendcounts[owner] += (slice_size / 4);
    }


    int * recvcounts = new int[size];
    int * recvdispls = new int[size];
    memset(recvdispls, 0, sizeof(int) * size);

    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    size_t send_size_total = sendcounts[0];
    size_t recv_size_total = recvcounts[0];
    for (int i=1; i<size; i++)
    {
        senddispls[i] = senddispls[i-1] + sendcounts[i-1];
        recvdispls[i] = recvdispls[i-1] + recvcounts[i-1];
        send_size_total += sendcounts[i];
        recv_size_total += recvcounts[i];
    }

    float * send_buf;
    CHECK_CUDA(cudaMalloc(&send_buf, sizeof(float) * send_size_total * 4));

    for (int64_t i=0; i<grid_dim; i++)
    {
        int owner = i + col_rank*grid_dim;
        int64_t offset = (int64_t)(senddispls[owner])*4;
        int64_t k_offset = i * slice_size;
        CHECK_CUDA(cudaMemcpyAsync(send_buf + offset,
                                  K + k_offset,
                                  sizeof(float) * slice_size,
                                  cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // reuse K pointer for recvbuf
    MPI_Alltoallv(send_buf, sendcounts, senddispls, MPI_LONG_DOUBLE,
                  K, recvcounts, recvdispls, MPI_LONG_DOUBLE,
                  MPI_COMM_WORLD);

    // Have to transpose my received tiles -- can trick cuBLAS into doing it
    float alpha = 1.0;
    float beta = 0.0;
    for (int64_t i=0; i<grid_dim; i++)
    {
        CHECK_CUBLAS(cublasSgeam(handle.dh(), CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 m/size, mb,
                                 &alpha, K + i*slice_size, mb,
                                 &beta,
                                 send_buf + i*slice_size, m/size,
                                 send_buf + i*slice_size, m/size));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::swap(send_buf, K);

    // Cleanup
    CHECK_CUDA(cudaFree(send_buf));

    delete[] sendcounts;
    delete[] senddispls;
    delete[] recvcounts;
    delete[] recvdispls;

    return K;

}

}  // namespace cpop
