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

#include "common.hh"

#include "popcorn/utils/utils.hh"
// #include "popcorn/kernel/polynomial_kernel.cuh"
#include "popcorn/kernel_matrix.hh"

using namespace popcorn;

cusparseSpMatDescr_t initialize_v(cusparseHandle_t& cusparse_handle, int m,
                                  int k) {
  std::vector<int> row;
  std::vector<int> col;
  std::vector<float> val;

  // TODO: Speed up initialization
  for (int c = 0; c < m; ++c) {
    int r = c % k;
    int l = (m / k) + ((r < m % k) ? 1 : 0);
    row.push_back(r);
    col.push_back(c);
    val.push_back(1.0f / l);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    for (int i = 0; i < m; ++i) {
      std::cout << "(" << row.at(i) << "," << col.at(i) << "," << val.at(i)
                << ")" << std::endl;
    }
  }

  void *cooRowInd, *cooColInd, *cooValues;

  cudaMalloc(&cooRowInd, m * sizeof(int));
  cudaMalloc(&cooColInd, m * sizeof(int));
  cudaMalloc(&cooValues, m * sizeof(float));

  cusparseSpMatDescr_t V;
  cusparseCreateCoo(&V, k, m, m, cooRowInd, cooColInd, cooValues,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // TODO: Move these somewhere else (can't free until after use)
  // cudaFree(cooRowInd);
  // cudaFree(cooColInd);
  // cudaFree(cooValues);

  return V;
}

cusparseDnMatDescr_t spmm(cusparseHandle_t& cusparse_handle,
                          cusparseSpMatDescr_t sparse,
                          cusparseDnMatDescr_t dense) {
  // Get input information
  int64_t sp_rows, sp_cols, dn_rows, dn_cols, ld;
  cudaDataType type;
  cusparseOrder_t order;
  cusparseSpMatGetSize(sparse, &sp_rows, &sp_cols, nullptr);
  cusparseDnMatGet(dense, &dn_rows, &dn_cols, &ld, nullptr, &type, &order);

  // Protect against bad input
  assert(sp_cols == dn_rows && "Inner dimension must be equal in size.");
  assert(type == CUDA_R_32F && "Matrix data must be FP32.");

  // Define constants
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Allocate memory for output
  float* out_data;
  cudaMalloc(&out_data, sp_rows * dn_cols * sizeof(float));
  cusparseDnMatDescr_t out;
  cusparseCreateDnMat(&out, sp_rows, dn_cols, ld, out_data, type, order);

  // Allocate workspace
  size_t buffer_size;
  void* buffer;
  cusparseSpMM_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse,
                          dense, &beta, out, CUDA_R_32F,
                          CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
  cudaMalloc(&buffer, buffer_size);

  // Perform SpMM
  cusparseSpMM(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse, dense, &beta,
               out, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);

  // Release workspace
  cudaFree(buffer);

  return out;
}

/**
 * Runs the distributed popcorn kernel k-means clustering algorithm.
 * Usage: srun popcorn [path] [m] [n] [k]
 *
 * [path] is the path to the dataset
 * [m] is the number of samples
 * [n] is the number of features
 * [k] is the number of clusters
 */
int main(int argc, char* argv[]) {
  assert(argc == 5 && "Invalid args. Must provide params [path] [m] [n] [k]");

#ifndef CUDA
  std::cout << "CUDA is unavailable. Exiting..." << std::endl;
  return 0;
#endif

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char* fpath = argv[1];
  int m = std::atoi(argv[2]);
  int n = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  wake_gpus(rank);

  cusparseHandle_t cusparse_handle;
  cusparseCreate(&cusparse_handle);

  // Load the original data with SLATE, this will be transposed
  auto PT = load_data(fpath, m, n);
  // slate::print("PT", *PT);

#ifdef P_BENCHMARK
  // Start the timer (after IO)
  auto start = hrc::now();
  double k_elapsed = 0;
  double v_elapsed = 0;
  double vk_elapsed = 0;
  double c_elapsed = 0;
  double d_elapsed = 0;
  double vr_elapsed = 0;
#endif

  // Compute the K matrix (internally, this will tranpose P and apply the kernel function)
  // TODO: make gamma, c, r as IO input
#ifdef P_BENCHMARK
  auto k_start = hrc::now();
#endif
  // auto poly_kernel = PolynomialKernel(1.0f, 1.0f, 1.0f);
  auto K = compute_kernel_matrix(PT);
  slate::print("K", *K);
  // TODO: Wrap local column grid of K data with cusparse descriptor
  cusparseDnMatDescr_t Kc;
#ifdef P_BENCHMARK
  k_elapsed += std::chrono::duration_cast<ms>(hrc::now() - k_start).count();
#endif

  if (rank == 0)
    std::cout << k_elapsed << std::endl;

// Initialize the V matrix
#ifdef P_BENCHMARK
  auto v_start = hrc::now();
#endif
  auto V = initialize_v(cusparse_handle, m, k);
#ifdef P_BENCHMARK
  v_elapsed += std::chrono::duration_cast<ms>(hrc::now() - v_start).count();
#endif
#ifdef P_DEBUG
  V.print("V");
#endif

  // Begin the main K means clustering loop
  for (int i = 0; i < 100; ++i) {

// Perform SpMM(VK)
#ifdef P_BENCHMARK
    auto vk_start = hrc::now();
#endif
    auto ET = spmm(cusparse_handle, V, Kc);
#ifdef P_BENCHMARK
    vk_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vk_start).count();
#endif
#ifdef P_DEBUG
    if (i == 0)
      ET.print("ET");
#endif

    // TODO: Clean up device data (inside mats)
    cusparseDestroySpMat(V);
    cusparseDestroyDnMat(ET);
    break;

    //       // Compute the centroid norms
    // #ifdef P_BENCHMARK
    //     auto c_start = hrc::now();
    // #endif
    //     auto C = initialize_cnorm(V, ET);
    // #ifdef P_BENCHMARK
    //     c_elapsed += std::chrono::duration_cast<ms>(hrc::now() - c_start).count();
    // #endif

    //     // Compute the D matrix
    // #ifdef P_BENCHMARK
    //     auto d_start = hrc::now();
    // #endif
    //     compute_d(ET, C);
    // #ifdef P_BENCHMARK
    //     d_elapsed += std::chrono::duration_cast<ms>(hrc::now() - d_start).count();
    // #endif

    //     // Reinitialize V matrix
    // #ifdef P_BENCHMARK
    //     auto vr_start = hrc::now();
    // #endif
    //     V = reinitialize_v(V, ET);
    // #ifdef P_BENCHMARK
    //     vr_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vr_start).count();
    // #endif
  }

  //   // Output cluster assignments
  //   std::string prefix = std::string(fpath);
  //   std::string suffix = "_out";
  //   save_assignments(V, (prefix + suffix).c_str());

  cusparseDestroyDnMat(Kc);
  cusparseDestroy(cusparse_handle);

  MPI_Finalize();
  return 0;
}
