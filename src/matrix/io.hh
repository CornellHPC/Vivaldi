#ifndef DISTRIBUTED_POPCORN_IO_H
#define DISTRIBUTED_POPCORN_IO_H

#include <mpi.h>

#include "../common.hh"
#include "matrix.hh"

namespace matrix {

// Loads matrix at filename with m rows and n columns
template <typename T>
slate::Matrix<T> load_slate_mat(const char *filename, int64_t m, int64_t n,
                                int64_t mb, int64_t nb, MPI_Comm comm) {
  // Open file
  MPI_File fh;
  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = m * n * sizeof(T);
  T *data = (T *)malloc(count);
  MPI_Datatype type = combblas::MPIType<T>();
  MPI_File_read_all(fh, data, m * n, type, MPI_STATUS_IGNORE);

  // Create matrix
  int p = square_grid_dim(comm);
  slate::Matrix<T> M(m, n, mb, nb, p, p, comm);
#ifdef CUDA
  M.insertLocalTiles(slate::Target::Devices);
#else
  M.insertLocalTiles(slate::Target::Host);
#endif

  // Initialize data
  fill_slate_mat_with_buffer(M, data);

  // Clean up
  free(data);
  MPI_File_close(&fh);

  return M;
}

} // namespace matrix

#endif // DISTRIBUTED_POPCORN_IO_H
