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

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  /** LOAD DATA */
  auto PT = load_matrix(fpath, m, n);
  // slate::print("PT", *PT);

  // Start the timer (after dataset IO)
  auto start = hrc::now();
  double k_elapsed, init_v_elapsed, vk_elapsed;

  /** COMPUTING K */
  auto k_start = hrc::now();
  auto K = compute_kernel_matrix(PT);
  K->tileLayoutConvertOnDevices(blas::Layout::RowMajor);
  // slate::print("K", *K);
  k_elapsed = std::chrono::duration_cast<ms>(hrc::now() - k_start).count();

  /** CREATE CUSPARSE DENSE MATRIX FROM K */
  // Each rank can have at most 2 pieces of K
  bool multiple = rank + size < K->nt();
  float *K1, *K2;
  cusparseDnMatDescr_t K1_desc, K2_desc;

  popcorn::extract_kernel_tiles(K1, K, rank);
  if (multiple)
    popcorn::extract_kernel_tiles(K2, K, rank + size);

  cusparseCreateDnMat(&K1_desc, K->m(), K->tileNb(rank), K->m(), K1, CUDA_R_32F,
                      CUSPARSE_ORDER_ROW);
  if (multiple)
    cusparseCreateDnMat(&K2_desc, K->m(), K->tileNb(rank + size), K->m(), K2,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

  auto v_start = hrc::now();

  // Initialize the V matrix
  auto V = initialize_v(handle, m, k);

  init_v_elapsed = std::chrono::duration_cast<ms>(hrc::now() - v_start).count();

  // Begin the main K means clustering loop
  // for (int i = 0; i < 100; ++i) {
  //   // Perform SpMM(VK)
  //   auto vk_start = hrc::now();
  //   // auto ET = spmm(cusparse_handle, V, Kc);
  //   vk_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vk_start).count();

  //   // TODO: Clean up device data (inside mats)
  //   cusparseDestroySpMat(V);
  //   // cusparseDestroyDnMat(ET);
  //   break;
  // }

  /** PRINT TIMES */
  if (rank == 0) {
    std::cout << "Time K: " << k_elapsed << "ms" << std::endl;
  }

  /** DESTROY */
  cusparseDestroyDnMat(K1_desc);
  if (multiple) cusparseDestroyDnMat(K2_desc);
  cudaFree(K1);
  if (multiple) cudaFree(K2);

  // cusparseDestroyDnMat(Kc);
  cusparseDestroy(handle);

  MPI_Finalize();
  return 0;
}
