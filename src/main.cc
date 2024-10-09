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
#include "common.hh"
#include "matrix/io.hh"
#include "matrix/kernel_matrix.hh"
#include "matrix/matrix.hh"
#include "popcorn/utils/utils.hh"

using namespace popcorn;

// Define CombBLAS semiring
template <typename UV>
using SR = combblas::PlusTimesSRing<UV, UV>;

// Drives the algorithm
void cluster(char *points_path, int m, int n, int k, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0)
#ifdef CUDA
    std::cout << "Running on: CUDA" << std::endl;
  wake_gpus(rank);
#else
    std::cout << "Running on: CPU" << std::endl;
#endif

  if (rank == 0) std::cout << "Reading data from " << points_path << std::endl;

  int p = square_grid_dim(comm);
  auto sP = matrix::load_slate_mat(points_path, m, n, tile_dim(comm, m),
                                   tile_dim(comm, n), comm);
  slate::print("P is", sP);
  // TODO: make gamma, c, r as IO input
  auto sK =
      matrix::slate_point_mat_to_polynomial_kernel_mat(sP, 1.0f, 1.0f, 2.0f);
  slate::print("K is ", sK);

  auto cK = matrix::slate_mat_to_combblas_dpm(sK);
  cK.PrintToFile("out/K");

  if (rank == 0) std::cout << "Wrote K to disc" << std::endl;

  auto cV =
      matrix::initialize_combblas_v_matrix(cK.getgnrow(), k, sK.mpiComm());
  cV.PrintInfo();

  combblas::spmm_stats stats;
  auto O = combblas::SpMM_sC<SR<DATA_TYPE>, int64_t, DATA_TYPE, DATA_TYPE,
                             UDER<DATA_TYPE>>(cV, cK, stats);
  O.PrintToFile("out/O");
}

// Handles command-line arguments
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  assert(argc == 5 &&
         "Must pass valid path to point data, number of rows, "
         "number of columns, and value of k.");

  char *points_path = argv[1];
  int m = std::atoi(argv[2]);
  int n = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  // TODO: Create subcommunicator here if necessary
  // Possible reasons include:
  //   1. Non-square number of processors
  //   2. More processors than pieces of data
  //
  // We will assume that all communicators provide a
  // square process grid going forward. The following
  // assertion can be removed once a subcommunicator
  // is properly created and passed.
  assert(is_square_grid(MPI_COMM_WORLD) &&
         "Must provide square number of ranks.");

  cluster(points_path, m, n, k, MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
