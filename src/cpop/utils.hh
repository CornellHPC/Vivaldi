#ifndef CPOP_UTILS_HH
#define CPOP_UTILS_HH

#include <chrono>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <mpi.h>

namespace cpop {

/**
 * @brief Setup GPUs
 * 
 * @param rank 
 */
void wake_gpus(int rank);

/**
 * @brief Calculates time delta
 * 
 * @param start Time right now from high resolution clock
 * @return Time elapsed in ms
 */
int64_t get_time_elapsed(std::chrono::_V2::system_clock::time_point start);

/**
 * @brief Print buffer on device
 * 
 * @param buf The device pointer to the buffer
 * @param count Number of elements in buf to print
 */
template <typename T>
void print_device_buffer(T* buf, size_t count) {
  T* temp = (T*)malloc(count * sizeof(T));
  cudaMemcpy(temp, buf, count * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < count; i++)
    std::cout << temp[i] << " ";
  std::cout << std::endl;

  free(temp);
}

}  // namespace cpop

#endif  // CPOP_UTILS_HH
