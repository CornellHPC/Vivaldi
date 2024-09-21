#include "CombBLAS/CombBLAS.h"
#include <cassert>
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

  assert(argc == 2 && "Must pass valid path to point data.");

  std::cout << "Reading data from " << argv[1] << std::endl;
  slate::Matrix<float> M = load_slate_mat<float>(argv[1], size, 4, 4);
  slate::print("M from main is", M);
  auto C = point_mat_to_polynomial_kernel_mat(M, 1.0f, 0.0f, 0.0f);
  slate::print("C from main is ", C);

  MPI_Finalize();
  return 0;
}
