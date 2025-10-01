#ifndef CPOP_UTILS_HH
#define CPOP_UTILS_HH
#include <vector>
#include "unistd.h"
#include <chrono>

using hrc = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;


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

#define CHECK_CUBLAS(func)                                                   \
  {                                                                          \
    cublasStatus_t status = (func);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      printf("CUBLAS API failed at line %d with error: %s (%d)\n", __LINE__, \
             cublasGetErrorString(status), status);                          \
    }                                                                        \
  }

#include <chrono>
#include <cmath>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <mpi.h>

namespace cpop {

const char* cublasGetErrorString(cublasStatus_t status);

struct ArgParse {
  std::string path;
  int m, n, k;
  bool s;
  float gamma, c, r;
  std::string output;
  std::string benchmark;
  int niter;
  int convergence;
  std::string alg;

  ArgParse(int argc, char* argv[]);
};

/// @brief Timer struct
struct Timer {
  // Non-IO overall elapsed time (basic mode)
  int64_t elapsed = 0;

  // IO
  int64_t io = 0;

  // K
  int64_t k_elapsed = 0;
  int64_t k_redist = 0;

  // Vi
  int64_t vi_elapsed = 0;

  // todo: allgather average over ranks
  // E
  int64_t e_elapsed = 0;
  int64_t e_transpose = 0;
  int64_t e_mpi = 0;
  int64_t e_reduce = 0;
  int64_t e_gather = 0;
  int64_t e_spmm = 0;
  int64_t e_other = 0;

  // Z
  int64_t z_elapsed = 0;

  // C
  int64_t c_elapsed = 0;
  int64_t c_mpi = 0;
  int64_t c_computation = 0;

  // Vr
  int64_t vr_elapsed = 0;
  int64_t vr_mpi = 0;
  int64_t vr_computation = 0;

  // Number of iterations before convergence
  int64_t niter = 0;
  int* dead_proc_counts = nullptr;

  // nnz per process for 2d
  int64_t * nnz_perproc = nullptr;
  int64_t * global_nnz_perproc = nullptr;


  void gather_nnz_perproc();

  /**
   * @brief Save only the elapsed time as a single value to a file.
   * 
   * @param path filename
   */
  void save_elapsed(const char* path);

  /**
   * @brief Save all benchmarked values to file.
   * 
   * @param path filename
   * @param score cluster score
   */
  void save_all(const char* path, float score);
};

class Handle {
  bool sparse;
  cusparseHandle_t s_handle;
  cublasHandle_t d_handle;

 public:
  Handle(bool sparse) {
    this->sparse = sparse;
    cusparseCreate(&s_handle);
    cublasCreate_v2(&d_handle);
  }

  ~Handle() {
    cusparseDestroy(s_handle);
    cublasDestroy_v2(d_handle);
  }

  bool isSparse() { return sparse; }

  cusparseHandle_t sh() { return s_handle; }

  cublasHandle_t dh() { return d_handle; }
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
 * @brief Return a newly-allocated array of tile sizes per process
 */
int* compute_tile_sizes(int m, int nprocs);


//std::vector<std::vector<std::array<int, 2>>> compute_tile_sizes2d(int m, int nprocs);


int square_grid_dim(MPI_Comm comm);
bool is_square_grid(MPI_Comm comm);
int tile_dim(MPI_Comm comm, int x);
int tile_dim2(int p, int x);
void print_phase(const char * name);
void print_line();

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

template <typename T>
void print_host_buffer(T* buf, size_t count, int rank) 
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout<<"Rank: "<<rank<<std::endl;
  for (int i = 0; i < count; ++i)
    std::cout << buf[i] << " ";
  std::cout << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int p=0; p<size; p++)
  {
      if (p==rank)
      {
          std::cout << "--Process "<<p<<"--"<<std::endl;
          for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
              std::cout << temp[i * w + j] << " ";
            }
            std::cout << "\n";
          }
          std::cout << std::flush;
      }
      MPI_Barrier(MPI_COMM_WORLD);
  }

  free(temp);
}


template <typename... Args>
void par_print(const char * str, Args... args)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stdout, "---Process %d---\n", rank);
    fprintf(stdout, str, args...);
    fprintf(stdout, "----------------\n");
    fflush(stdout);
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);
}

extern Timer timer;

}  // namespace cpop

#endif  // CPOP_UTILS_HH
