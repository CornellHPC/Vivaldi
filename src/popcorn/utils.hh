#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>

#include <mpi.h>
#include <cuda_runtime.h>

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
 * @param rank_to_print Rank to print
 */
void print_device_buffer(float* buf, size_t count, int rank_to_print);

/**
 * @brief Calculates time delta
 * 
 * @param start Time right now from high resolution clock
 * @return Time elapsed in ms
 */
int64_t get_time_elapsed(std::chrono::_V2::system_clock::time_point start);

}