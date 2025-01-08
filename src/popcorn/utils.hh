#include <mpi.h>

namespace popcorn {

/**
 * @brief Simple test to make sure GPUs are functioning
 * 
 * @param rank 
 */
void wake_gpus(int rank);

template <typename F>
auto time_function(F f, double* elapsed);

}