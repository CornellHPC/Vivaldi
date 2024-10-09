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
#include "popcorn/utils/utils.hh"
#include "popcorn/cluster.hh"

using namespace popcorn;

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
