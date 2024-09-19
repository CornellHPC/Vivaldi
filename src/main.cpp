#include "CombBLAS/CombBLAS.h"
// Macro "Error" is defined in CombBLAS and SLATE. It is unused in
// CombBLAS, so we undefine it here to prevent name collisions.
#undef Error

#include "mpi.h"
#include "mpio.h"
#include "slate/slate.hh"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// Loads matrix at filename with m rows and n columns
slate::Matrix<float> load_matrix(const char *filename, int64_t m, int64_t n,
                                 int64_t mb = 2, int64_t nb = 2) {
  // Get communicator information
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int p = std::sqrt(size);

  // Open file
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  // Read data
  float *buf = (float *)malloc(m * n * sizeof(float));
  MPI_File_read_all(fh, buf, m * n, MPI_FLOAT, MPI_STATUS_IGNORE);

  // Create matrix
  slate::Matrix<float> M(m, n, mb, nb, p, p, MPI_COMM_WORLD);
  M.insertLocalTiles();

  // Initialize data
  for (int64_t j = 0; j < M.nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) { // j loops over block rows
      if (M.tileIsLocal(i, j)) {
        slate::Tile<float> tile = M(i, j);
        int64_t lda = tile.stride();
        float *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {   // jj loops over columns
          for (int64_t ii = 0; ii < tile.mb(); ++ii) { // ii loops over rows
            int64_t global_row = i * mb + ii;
            int64_t global_column = j * nb + jj;
            A[ii + jj * lda] = buf[global_row + global_column * m];
          }
        }
      }
    }
  }

  // Clean up
  free(buf);
  MPI_File_close(&fh);

  return M;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  slate::Matrix<float> M = load_matrix(
      "/global/homes/m/mrrubino/cpp/distributed-popcorn/test", 4, 4);
  slate::print("M", M);

  MPI_Finalize();
  return 0;
}
