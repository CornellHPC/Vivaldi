#include "io.hh"

slate::Matrix<DATA_TYPE> matrix::load_slate_mat(const char *filename,
                                                int mpi_size, int64_t m,
                                                int64_t n) {
  // Get communicator information
  int p, q;
  grid_size(mpi_size, &p, &q);

  // Open file
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = m * n * sizeof(DATA_TYPE);
  DATA_TYPE *data = (DATA_TYPE *)malloc(count);
  MPI_File_read_all(fh, data, m * n, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create matrix
  slate::Matrix<DATA_TYPE> M(m, n, SLATE_TILE_M, SLATE_TILE_N, p, q,
                             MPI_COMM_WORLD);
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
