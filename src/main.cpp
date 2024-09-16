#include "CombBLAS/CombBLAS.h"
#include <cstdint>
// Macro "Error" is defined in CombBLAS and SLATE. It is unused in
// CombBLAS, so we undefine it here to prevent name collisions.
#undef Error

#include "slate/slate.hh"

#include <iostream>
#include <mpi.h>

// Column-major indexing
template <typename scalar_type>
void random_matrix(int64_t m, int64_t n, scalar_type *A, int64_t lda) {
  for (int64_t j = 0; j < n; ++j) {
    for (int64_t i = 0; i < m; ++i) {
      A[i + j * lda] = blas::make_scalar<scalar_type>(
          rand() / double(RAND_MAX), rand() / double(RAND_MAX));
    }
  }
}

// Column-major indexing
template <typename matrix_type> void random_matrix(matrix_type &A) {
  for (int64_t j = 0; j < A.nt(); ++j) {
    for (int64_t i = 0; i < A.mt(); ++i) {
      if (A.tileIsLocal(i, j)) {
        try {
          auto T = A(i, j);
          random_matrix(T.mb(), T.nb(), T.data(), T.stride());
        } catch (...) {
          // ignore missing tiles
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int rank, nprocs;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  float alpha = 1.0f, beta = 0.0f;
  int64_t m = 16, n = 16, k = 16, nb = 2;

  slate::Matrix<float> A(m, n, nb, 2, 2, MPI_COMM_WORLD);
  slate::Matrix<float> B(m, n, nb, 2, 2, MPI_COMM_WORLD);
  slate::Matrix<float> C(m, n, nb, 2, 2, MPI_COMM_WORLD);
  A.insertLocalTiles();
  B.insertLocalTiles();
  C.insertLocalTiles();
  random_matrix(A);
  random_matrix(B);
  random_matrix(C);

  slate::gemm(alpha, A, B, beta, C);

  MPI_Finalize();

  std::cout << "Graceful exit. Yay!" << std::endl;
  return 0;
}

/*

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(MPI_COMM_WORLD, "../DryBeanDataset/Dry_Bean_Dataset.arff",
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  constexpr int num_chars = 10;
  char *buf = (char *)malloc(num_chars * sizeof(char));
  MPI_File_read(fh, buf, num_chars, MPI_CHAR, &status);
  MPI_File_close(&fh);

  std::cout << "Rank " << rank << " received ";
  for (int i = 0; i < num_chars; ++i)
    std::cout << buf[i];
  std::cout << std::endl;
*/
