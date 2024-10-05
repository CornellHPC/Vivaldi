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
#include "util.hh"

// Define CombBLAS semiring
template <typename UV> using SR = combblas::PlusTimesSRing<UV, UV>;

// Drives the algorithm
void cluster(char *points_path, int m, int n, int k, int rank, int size) {
  if (rank == 0)
#ifdef CUDA
    std::cout << "Running on: CUDA" << std::endl;
#else
    std::cout << "Running on: CPU" << std::endl;
#endif

#ifdef CUDA
  wake_gpus(rank);
#endif

  if (rank == 0)
    std::cout << "Reading data from " << points_path << std::endl;

  auto sP = matrix::load_slate_mat(points_path, size, m, n);
  std::cout << "Rank: " << rank << " at line 42" << std::endl;
  slate::print("P is", sP);
  // TODO: make gamma, c, r as IO input
  auto sK =
      matrix::slate_point_mat_to_polynomial_kernel_mat(sP, 1.0f, 1.0f, 2.0f);
  slate::print("K is ", sK);

  if (rank == 0)
    std::cout << sK.m() << " " << sK.n() << " " << sK.mt() << " " << sK.nt()
              << " " << sK.tileMb(0) << " " << sK.tileNb(0);

  auto cK = matrix::slate_mat_to_combblas_dpm(sK);
  cK.PrintToFile("out/K");
  std::cout << "Rank: " << rank << " at line 50" << std::endl;

  if (rank == 0)
    std::cout << "Wrote K to disc" << std::endl;

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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(argc == 5 && "Must pass valid path to point data, number of rows, "
                      "number of columns, and value of k.");
  cluster(argv[1], std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
          rank, size);

  MPI_Finalize();
  return 0;
}
