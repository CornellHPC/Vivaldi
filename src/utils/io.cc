#include "io.hh"

slate::Matrix<DATA_TYPE> load_slate_mat(const char *filename, int mpi_size,
                                        int64_t m, int64_t n, int64_t mb,
                                        int64_t nb) {
  // Get communicator information
  int p;
  grid_size(mpi_size, &p, &p);

  // Open file
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  size_t count = m * n * sizeof(DATA_TYPE);
  DATA_TYPE *data = (DATA_TYPE *)malloc(count);
  MPI_File_read_all(fh, data, m * n, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Copy to device
  DATA_TYPE *buf;
#ifdef CUDA
  cudaMalloc(&buf, count);
  cudaMemcpy(buf, data, count, cudaMemcpyHostToDevice);
#else
  buf = data;
#endif

  // Create matrix
  slate::Matrix<DATA_TYPE> M(m, n, mb, nb, p, p, MPI_COMM_WORLD);
#ifdef CUDA
  M.insertLocalTiles(slate::Target::Devices);
#else
  M.insertLocalTiles(slate::Target::Host);
#endif

  // Initialize data
  fill_slate_mat_with_buffer(M, buf);

  // Clean up
  free(data);
#ifdef CUDA
  cudaFree(buf);
#endif
  MPI_File_close(&fh);

  return M;
}