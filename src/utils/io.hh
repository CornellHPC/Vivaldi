#ifndef DISTRIBUTED_POPCORN_IO_H
#define DISTRIBUTED_POPCORN_IO_H

#include <mpi.h>

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
  scalar_type *buf = (scalar_type *)malloc(m * n * sizeof(scalar_type));
  MPI_File_read_all(fh, buf, m * n, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create matrix
  slate::Matrix<scalar_type> M(m, n, mb, nb, p, p, MPI_COMM_WORLD);
  M.insertLocalTiles();

  // Initialize data
  fill_mat_with_buffer(M, buf);

  // Clean up
  free(buf);
  MPI_File_close(&fh);

  return M;
}

#endif // DISTRIBUTED_POPCORN_IO_H
