#ifndef DISTRIBUTED_POPCORN_IO_H
#define DISTRIBUTED_POPCORN_IO_H

#include <mpi.h>

#include "cuda.hh"
#include "matrix.hh"
#include "slate/slate.hh"

// Loads matrix at filename with m rows and n columns
template <typename scalar_type>
slate::Matrix<scalar_type> load_slate_mat(const char *filename, int mpi_size,
                                          int64_t m, int64_t n, int64_t mb = 2,
                                          int64_t nb = 2) {
  // Get communicator information
  int p;
  grid_size(mpi_size, &p, &p);

  // Open file
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = m * n * sizeof(scalar_type);
  scalar_type *data = (scalar_type *)malloc(count);
  MPI_File_read_all(fh, data, m * n, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Copy to device
  scalar_type *buf;
  if (CUDA_AVAILABLE) {
    buf = cudaMalloc(&buf, count));
    cudaMemcpy(buf, data, count, cudaMemcpyHostToDevice);
  } else {
    buf = data;
  }

  // Create matrix
  slate::Matrix<scalar_type> M(m, n, mb, nb, p, p, MPI_COMM_WORLD);
  M.insertLocalTiles(CUDA_AVAILABLE ? slate::Target::Devices
                                    : slate::Target::Host);

  // Initialize data
  fill_slate_mat_with_buffer(M, buf);

  // Clean up
  free(data);
  if (CUDA_AVAILABLE)
    cudaFree(buf);
  MPI_File_close(&fh);

  return M;
}

#endif // DISTRIBUTED_POPCORN_IO_H
