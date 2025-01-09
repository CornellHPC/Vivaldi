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

#include "popcorn/utils.hh"
#include "popcorn/compute_kernel.hh"
#include "popcorn/compute_sparse.hh"

#define P_BENCHMARK

using namespace popcorn;

using hrc = std::chrono::high_resolution_clock;
using s = std::chrono::seconds;
using ms = std::chrono::milliseconds;

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
  // if (rank == 0) {
  //   for (int i = 0; i < m; ++i) {
  //     std::cout << "(" << row.at(i) << "," << col.at(i) << "," << val.at(i)
  //               << ")" << std::endl;
  //   }
  // }

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
  int64_t sp_rows, sp_cols, nnz, dn_rows, dn_cols, ld;
  cudaDataType type;
  cusparseOrder_t order;
  cusparseSpMatGetSize(sparse, &sp_rows, &sp_cols, &nnz);
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
 * @param path path to the dataset
 * @param m number of samples in the dataset
 * @param n number of features
 * @param k number of clusters to form
 */
int main(int argc, char* argv[]) {
  assert(argc == 5 && "Invalid args. Must provide params [path] [m] [n] [k]");

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

  // Load the original data with SLATE, this will be transposed
  auto PT = load_matrix(fpath, m, n);
  // slate::print("PT", *PT);

  // Start the timer (after dataset IO)
  auto start = hrc::now();
  double k_elapsed, v_elapsed, vk_elapsed;

  /** COMPUTING K */
  auto k_start = hrc::now();
  auto K = compute_kernel_matrix(PT);
  // slate::print("K", *K);
  k_elapsed += std::chrono::duration_cast<ms>(hrc::now() - k_start).count();
  
  /** CREATE CUSPARSE DENSE MATRIX FROM K */
  // bool multiple = false;
  // float *K1, *K2;
  // cusparseDnMatDescr_t K1_desc, K2_desc;
  // popcorn::create_kernel_descriptors(K1_desc, K2_desc, K1, K2, K, &multiple);

  

  
  auto v_start = hrc::now();

  // Initialize the V matrix
  auto V = initialize_v(handle, m, k);

  v_elapsed += std::chrono::duration_cast<ms>(hrc::now() - v_start).count();

  // Begin the main K means clustering loop
  for (int i = 0; i < 100; ++i) {
    // Perform SpMM(VK)
    auto vk_start = hrc::now();
    // auto ET = spmm(cusparse_handle, V, Kc);
    vk_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vk_start).count();

    // TODO: Clean up device data (inside mats)
    cusparseDestroySpMat(V);
    // cusparseDestroyDnMat(ET);
    break;
  }

  // cusparseDestroyDnMat(K1_desc);
  // if (multiple) cusparseDestroyDnMat(K2_desc);
  // cudaFree(K1);
  // if (multiple) cudaFree(K2);

  // cusparseDestroyDnMat(Kc);
  cusparseDestroy(handle);

  if (rank == 0) std::cout << "K took " << k_elapsed << "ms" << std::endl;

  MPI_Finalize();
  return 0;
}
