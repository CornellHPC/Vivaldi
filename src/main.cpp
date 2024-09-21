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

// Drives the algorithm
void cluster(char *points_path, int k, int rank, int size) {
  if (rank == 0)
    std::cout << "Reading data from " << points_path << std::endl;

  auto sP = load_slate_mat<float>(points_path, size, 4, 4);
  slate::print("P is", sP);
  auto sK = point_mat_to_polynomial_kernel_mat(sP, 1.0f, 0.0f, 0.0f);
  slate::print("K is ", sK);

  auto cK = slate_mat_to_combblas_dpm(sK);
  cK.PrintToFile("out/K");

  if (rank == 0)
    std::cout << "Wrote K to disc" << std::endl;
}

// Handles command-line arguments
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(argc == 3 && "Must pass valid path to point data and value of k.");
  cluster(argv[1], std::atoi(argv[2]), rank, size);

  MPI_Finalize();
  return 0;
}
