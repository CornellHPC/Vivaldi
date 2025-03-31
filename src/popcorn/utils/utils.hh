#ifndef UTILS_HH
#define UTILS_HH

#include "mpi.h"

#include <chrono>

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

int64_t get_time_elapsed(std::chrono::_V2::system_clock::time_point start);

}  // namespace popcorn

#endif  // UTILS_HH
