#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "mpi.h"
#include "mpio.h"

#include "common.hh"

#include "popcorn/utils/utils.hh"
// #include "popcorn/kernel/polynomial_kernel.cuh"
#include "popcorn/kernel_matrix.hh"

using namespace popcorn;

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
  return;
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
#ifdef P_BENCHMARK
  k_elapsed += std::chrono::duration_cast<ms>(hrc::now() - k_start).count();
#endif

  if (rank == 0)
    std::cout << k_elapsed << std::endl;

  //   // Initialize the V matrix
  // #ifdef P_BENCHMARK
  //   auto v_start = hrc::now();
  // #endif
  //   auto V = initialize_v(m, k, MPI_COMM_WORLD);
  // #ifdef P_BENCHMARK
  //   v_elapsed += std::chrono::duration_cast<ms>(hrc::now() - v_start).count();
  // #endif
  // #ifdef P_DEBUG
  //   V.print("V");
  // #endif

  //   // Convert K to CombBLAS
  //   auto cK = to_combblas(K);

  //   // Begin the main K means clustering loop
  //   for (int i = 0; i < 100; ++i) {

  //     // Perform SpMM(VK)
  // #ifdef P_BENCHMARK
  //     auto vk_start = hrc::now();
  // #endif
  //     auto ET = spmm(V, cK);
  // #ifdef P_BENCHMARK
  //     vk_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vk_start).count();
  // #endif
  // #ifdef P_DEBUG
  //     if (i == 0)
  //       ET.print("ET");
  // #endif

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
  //   }

  //   // Output cluster assignments
  //   std::string prefix = std::string(fpath);
  //   std::string suffix = "_out";
  //   save_assignments(V, (prefix + suffix).c_str());

  MPI_Finalize();
  return 0;
}
