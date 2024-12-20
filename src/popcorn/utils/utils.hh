#ifndef UTILS_HH
#define UTILS_HH

// Library imports
#include <mpi.h>

// Local imports
#include "../../common.hh"

namespace popcorn {

/**
 * @brief Simple test to make sure GPUs are functioning
 * 
 * @param myrank 
 */
void wake_gpus(int myrank);

int square_grid_dim(MPI_Comm comm);

bool is_square_grid(MPI_Comm comm);

int tile_dim(MPI_Comm comm, int x);

}  // namespace popcorn

#endif  // UTILS_HH
