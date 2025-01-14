#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "cusparse_v2.h"
#include "mpi.h"
#include "mpio.h"

#include "popcorn/compute_kernel.hh"
#include "popcorn/compute_sparse.hh"
#include "popcorn/utils.hh"

#define P_BENCHMARK

using namespace popcorn;

using hrc = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

/**
 * Runs the distributed popcorn kernel k-means clustering algorithm.
 * Usage: srun popcorn [path] [m] [n] [k]
 *
 * @param path path to the dataset
 * @param m number of samples in the dataset
 * @param n number of features
 * @param k number of clusters to form
 */
int main(int argc, char* argv[]) {
  assert(argc == 5 && "Invalid args. Must provide params [path] [m] [n] [k]");

  /** INITS */
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char* fpath = argv[1];
  int m = std::atoi(argv[2]);
  int n = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  wake_gpus(rank);
  slate::gpu_aware_mpi(false);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  /** LOAD DATA */
  auto PT = load_matrix(fpath, m, n, rank, size);
  // slate::print("PT", PT);

  /** START THE TIMER (after dataset IO) */
  auto start = hrc::now();

  /** COMPUTING K */
  auto k_start = hrc::now();

  auto K = compute_kernel_matrix(PT, rank, size);
  // slate::print("K", K);

  auto k_elapsed = get_time_elapsed(k_start);

  /** CREATE CUSPARSE DENSE MATRIX FROM K */
  auto slate_to_cusparse_start = hrc::now();

  // Each rank can have at most 2 pieces of K
  bool multiple = rank + size < K.nt();
  float *K1, *K2;
  cusparseDnMatDescr_t K1_desc, K2_desc;

  // Get pointer to actual matrix data
  K.tileLayoutConvertOnDevices(blas::Layout::RowMajor);
  popcorn::extract_kernel_tiles(&K1, K, rank);
  if (multiple) popcorn::extract_kernel_tiles(&K2, K, rank + size);

  // Create cuSPARSE dense matrix descriptors
  cusparseCreateDnMat(&K1_desc, K.m(), K.tileNb(rank), K.tileNb(rank), K1, CUDA_R_32F,
                      CUSPARSE_ORDER_ROW);
  if (multiple)
    cusparseCreateDnMat(&K2_desc, K.m(), K.tileNb(rank + size), K.tileNb(rank + size), K2,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

  auto slate_to_cusparse_elapsed =
      get_time_elapsed(slate_to_cusparse_start);

  /** INITIALIZE V */
  auto init_v_start = hrc::now();

  auto V = initialize_v(handle, m, k);

  auto init_v_elapsed = get_time_elapsed(init_v_start);

  /** K MEANS CLUSTERING LOOP */
  int niter = 1;
  for (int i = 0; i < niter; ++i) {
    /** SPMM ET = VK */
    auto ET_desc = popcorn::spmm(handle, V, K1_desc);

    float* vals;
    // cusparseDnMatGetValues(ET_desc, (void**)&vals);

    int64_t r, c, ld;

    // cusparseDnMatDescr_t dnMatDescr,
    // int64_t* rows,
    // int64_t* cols,
    // int64_t* ld,
    // void** values,
    cudaDataType type;
    cusparseOrder_t order;
    cusparseDnMatGet(ET_desc, &r, &c, &ld, (void**)&vals, &type, &order);
    if (rank == 0) std::cout << r << " " << c << " " << ld << " " << type << " " << order << std::endl;
 

    print_device_buffer_float(vals, r * c, 0);

    // auto vk_start = hrc::now();
    // auto ET = spmm(cusparse_handle, V, Kc);
    // vk_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vk_start).count();

    // TODO: Clean up device data (inside mats)
    cusparseDestroySpMat(V);
    cusparseDestroyDnMat(ET_desc);
    // break;
  }

  /** PRINT TIMES */
  if (rank == 0) {
    std::cout << "Time K: " << k_elapsed << "ms" << std::endl;
    std::cout << "Time SLATE to CUSPARSE: " << slate_to_cusparse_elapsed << "ms" << std::endl;
    std::cout << "Time Init V: " << init_v_elapsed << "ms" << std::endl;
  }

  /** DESTROY */
  cusparseDestroyDnMat(K1_desc);
  if (multiple) cusparseDestroyDnMat(K2_desc);
  cudaFree(K1);
  if (multiple) cudaFree(K2);
  
  cusparseDestroy(handle);

  MPI_Finalize();
  return 0;
}
