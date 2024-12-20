// C++ standard imports
#include <chrono>
#include <cstdlib>
#include <fstream>

// Local imports
#include "cluster.hh"
// #include "kernel/dist_kernel.cuh"
#include "kernel/polynomial_kernel.cuh"
// #include "mat/dense_mat.hh"
// #include "mat/sparse_mat.hh"
#include "utils/utils.hh"

#include "mat/cpop_blas.hh"
#include "mat/cpop_slate.hh"

namespace popcorn {

void cluster(char* data_path, int m, int n, int k, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

#ifndef CUDA
  if (rank == 0)
    std::cout << "CUDA is unavailable. Exiting..." << std::endl;
  return;
#endif

  wake_gpus(rank);
  // Cull points to make matrix evenly divisible
  int grid_dim = square_grid_dim(comm);
  m -= m % grid_dim;

#ifdef P_DEBUG
  if (rank == 0)
    std::cout << "Reading data from " << data_path << std::endl;
#endif

  // Load the original data with SLATE, this will be transposed
  auto PT = load_from_file(data_path, m, n, comm);
#ifdef P_DEBUG
  PT.print("PT");
#endif

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
  auto poly_kernel = PolynomialKernel(1.0f, 1.0f, 1.0f);
  auto K = compute_k(PT, poly_kernel);
#ifdef P_BENCHMARK
  k_elapsed += std::chrono::duration_cast<ms>(hrc::now() - k_start).count();
#endif
#ifdef P_DEBUG
  K.print("K");
#endif

  // Initialize the V matrix
#ifdef P_BENCHMARK
  auto v_start = hrc::now();
#endif
  auto V = initialize_v(m, k, comm);
#ifdef P_BENCHMARK
  v_elapsed += std::chrono::duration_cast<ms>(hrc::now() - v_start).count();
#endif
#ifdef P_DEBUG
  V.print("V");
#endif

  // Convert K to CombBLAS
  auto cK = to_combblas(K);

  // Begin the main K means clustering loop
  for (int i = 0; i < 100; ++i) {

    // Perform SpMM(VK)
#ifdef P_BENCHMARK
    auto vk_start = hrc::now();
#endif
    auto ET = spmm(V, cK);
#ifdef P_BENCHMARK
    vk_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vk_start).count();
#endif
#ifdef P_DEBUG
    if (i == 0)
      ET.print("ET");
#endif

      // Compute the centroid norms
#ifdef P_BENCHMARK
    auto c_start = hrc::now();
#endif
    auto C = initialize_cnorm(V, ET);
#ifdef P_BENCHMARK
    c_elapsed += std::chrono::duration_cast<ms>(hrc::now() - c_start).count();
#endif

    // Compute the D matrix
#ifdef P_BENCHMARK
    auto d_start = hrc::now();
#endif
    compute_d(ET, C);
#ifdef P_BENCHMARK
    d_elapsed += std::chrono::duration_cast<ms>(hrc::now() - d_start).count();
#endif

    // Reinitialize V matrix
#ifdef P_BENCHMARK
    auto vr_start = hrc::now();
#endif
    V = reinitialize_v(V, ET);
#ifdef P_BENCHMARK
    vr_elapsed += std::chrono::duration_cast<ms>(hrc::now() - vr_start).count();
#endif
  }

#ifdef P_BENCHMARK
  // Stop the timer (before IO)
  auto end = hrc::now();

  // Output runtime
  double elapsed = std::chrono::duration_cast<ms>(end - start).count();
  if (rank == 0) {
    std::cout << "Runtime: " << elapsed << " ms" << std::endl;
    std::cout << "K: " << k_elapsed << " ms" << std::endl;
    std::cout << "V: " << v_elapsed << " ms" << std::endl;
    std::cout << "VK: " << vk_elapsed << " ms" << std::endl;
    std::cout << "C: " << c_elapsed << " ms" << std::endl;
    std::cout << "D: " << d_elapsed << " ms" << std::endl;
    std::cout << "V (re): " << vr_elapsed << " ms" << std::endl;
  }
#endif

  // Output cluster assignments
  std::string prefix = std::string(data_path);
  std::string suffix = "_out";
  save_assignments(V, (prefix + suffix).c_str());
}

}  // namespace popcorn
