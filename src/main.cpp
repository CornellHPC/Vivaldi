#include "CombBLAS/CombBLAS.h"
// Macro "Error" is defined in CombBLAS and SLATE. It is unused in
// CombBLAS, so we undefine it here to prevent name collisions.
#undef Error

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "mpio.h"
#include "slate/slate.hh"
#include "utils/io.hh"
#include "utils/matrix.hh"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  slate::Matrix<float> M = load_matrix<float>(
      "/global/homes/n/npi2/distributed-popcorn/test", size, 4, 4);
  slate::Matrix<float> C(4, 4, 2, 2, 2, MPI_COMM_WORLD);
  C.insertLocalTiles();
  construct_kernel_matrix_with_gemm(M, C, 1.0f, 0.0f, 0.0f);
  slate::print("C from main is ", C);
  MPI_Finalize();
  return 0;
}
