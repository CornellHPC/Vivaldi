#ifndef CPOP_UTILS_HH
#define CPOP_UTILS_HH

#include <chrono>
#include <cmath>

#include <cuda_runtime.h>
#include <mpi.h>

#include "gpu_kernels.cuh"

namespace cpop {

/**
 * @brief Setup GPUs
 * 
 * @param rank 
 */
void wake_gpus(int rank);

/**
 * @brief Print float buffer on device
 * 
 * @param buf
 * @param count Number of elements in buf
 * @param rank_to_print Rank to print
 */
void print_device_buffer_float(float* buf, size_t count, int rank_to_print);

/**
 * @brief Print int buffer on device
 * 
 * @param buf
 * @param count Number of elements in buf
 * @param rank_to_print Rank to print
 */
void print_device_buffer_int(int* buf, size_t count, int rank_to_print);

/**
 * @brief Calculates time delta
 * 
 * @param start Time right now from high resolution clock
 * @return Time elapsed in ms
 */
int64_t get_time_elapsed(std::chrono::_V2::system_clock::time_point start);

}  // namespace cpop

#endif  // CPOP_UTILS_HH
