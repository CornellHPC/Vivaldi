#include <iostream>
#include <mpi.h>

#include "gpu_kernels.cuh"

namespace popcorn {

/**
 * @brief Simple test to make sure GPUs are functioning
 * 
 * @param rank 
 */
void wake_gpus(int rank);

/**
 * @brief Print buffer on device
 * 
 * @param buf
 * @param count Number of elements in buf
 */
void print_device_buffer(float* buf, size_t count);

}