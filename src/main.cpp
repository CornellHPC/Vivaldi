// CombBLAS assumes this is available
using namespace std;

// C++ standard imports
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// Library imports
#include "mpi.h"
#include "mpio.h"

// Local imports
#include "kernel/kernel_matrix.hh"
#include "utils/io.hh"
#include "utils/matrix.hh"
#include "common.hh"

// Define CombBLAS semiring
template <typename UV> using SR = combblas::PlusTimesSRing<UV, UV>;

// Drives the algorithm
void cluster(char *points_path, int k, int rank, int size) {
  if (rank == 0) std::cout << "Running on: ";
#ifdef CUDA
  std::cout << "CUDA" << std::endl;
#else
  std::cout << "CPU" << std::endl;
#endif

  if (rank == 0) std::cout << "Reading data from " << points_path << std::endl;

  auto sP = load_slate_mat(points_path, size, 4, 4);
  slate::print("P is", sP);
  auto sK = slate_point_mat_to_polynomial_kernel_mat(sP, 1.0f, 1.0f, 2.0f);
  slate::print("K is ", sK);

  auto cK = slate_mat_to_combblas_dpm(sK);
  cK.PrintToFile("out/K");

  if (rank == 0) std::cout << "Wrote K to disc" << std::endl;

  auto cV = initialize_combblas_v_matrix(cK);
  cV.PrintInfo();
  // cV.Dump("out/V");

  combblas::spmm_stats stats;
  auto O = combblas::SpMM_sC<SR<DATA_TYPE>, int64_t, DATA_TYPE, DATA_TYPE,
                             UDER<DATA_TYPE>>(cV, cK, stats);
  O.PrintToFile("out/O");
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
