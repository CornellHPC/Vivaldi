#ifndef DISTRIBUTED_POPCORN_CLUSTER_HH
#define DISTRIBUTED_POPCORN_CLUSTER_HH

// Library imports
#include <mpi.h>

// Local imports
#include "../common.hh"

namespace popcorn {

/**
 * @brief Drives the distributed clustering algorithm
 *
 * @param data_path Path to data file
 * @param m Number of rows in data
 * @param n Number of cols in data
 * @param k Number of clusters to form
 * @param comm MPI communicator
 */
void cluster(char* data_path, int m, int n, int k, MPI_Comm comm);

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_CLUSTER_HH
