#ifndef CPOP_UTILS_HH
#define CPOP_UTILS_HH

#define CHECK_CUDA(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
    }                                                                      \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
    }                                                                          \
  }

#include <chrono>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <mpi.h>

namespace cpop {

struct ArgParse {
  std::string path;
  int m, n, k;
  float gamma, c, r;
  std::string output;
  int niter;
  bool convergence;

  ArgParse(int argc, char* argv[]); 
};

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

  for (int i = 0; i < count; ++i)
    std::cout << temp[i] << " ";
  std::cout << std::endl;

  free(temp);
}

/**
 * @brief Print matrix on device
 *
 * @param mat The device pointer to the matrix
 * @param h The height of the matrix
 * @param w The width of the matrix
 */
template <typename T>
void print_device_matrix(T* mat, size_t h, size_t w) {
  T* temp = (T*)malloc(h * w * sizeof(T));
  cudaMemcpy(temp, mat, h * w * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; i < w; ++j) {
      std::cout << temp[i * w + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::flush;

  free(temp);
}

}  // namespace cpop

#endif  // CPOP_UTILS_HH
