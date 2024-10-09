#ifndef DISTRIBUTED_POPCORN_CLUSTER_HH
#define DISTRIBUTED_POPCORN_CLUSTER_HH

#include "../common.hh"

#include <mpi.h>

namespace popcorn {

/**
 * @brief Drives the algorithm
 * 
 * @param points_path 
 * @param m 
 * @param n 
 * @param k 
 * @param comm 
 */
void cluster(char *points_path, int m, int n, int k, MPI_Comm comm);

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_CLUSTER_HH