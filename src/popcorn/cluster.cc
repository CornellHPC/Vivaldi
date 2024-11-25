// C++ standard imports
#include <chrono>
#include <cstdlib>

// Local imports
#include "cluster.hh"
#include "kernel/dist_kernel.cuh"
#include "kernel/polynomial_kernel.cuh"
#include "mat/dense_mat.hh"
#include "mat/sparse_mat.hh"
#include "utils/utils.hh"

namespace popcorn {

void cluster(char* data_path, int m, int n, int k, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

#ifdef P_DEBUG
  if (rank == 0)
#ifdef CUDA
    std::cout << "Running on: CUDA" << std::endl;
#else
    std::cout << "CUDA is unavailable. Some things may not work properly."
              << std::endl;
#endif
#endif

#ifdef CUDA
  wake_gpus(rank);
#endif

  // Cull points to make matrix evenly divisible
  int grid_dim = square_grid_dim(comm);
  m -= m % grid_dim;

#ifdef P_DEBUG
  if (rank == 0)
    std::cout << "Reading data from " << data_path << std::endl;
#endif

  // Load the original data with SLATE, this will be transposed
  auto PT = DenseMat::load_from_file(data_path, m, n, comm);
#ifdef P_DEBUG
  // PT.print("PT");
#endif

#ifdef P_BENCHMARK
  // Start the timer (after IO)
  auto start = std::chrono::high_resolution_clock::now();
#endif

  // Transpose back to obtain the original matrix
  auto P = PT.transpose();
  // P.print("P");

  // Compute the K matrix
  // TODO: make gamma, c, r as IO input
  auto K = P.gemm(PT);
  auto poly_kernel = PolynomialKernel(1.0f, 1.0f, 1.0f);
  K.apply(poly_kernel);
#ifdef P_DEBUG
  // K.print("K");
#endif

  // Initialize the V matrix
  auto V = SparseMat::initialize_v(m, k, comm);
#ifdef P_DEBUG
  V.print("V");
#endif

  for (int i = 0; i < 100; ++i) {
    // Perform SpMM(VK)
    auto ET = V.spmm(K);
#ifdef P_DEBUG
    if (i == 0)
      ET.print("ET");
#endif

    // Compute the centroid norms
    auto C = DenseMat::initialize_cnorm(V, ET);

    // Compute the D matrix
    auto dist_kernel = DistKernel();
    auto D = dist_kernel.kernel(ET.rows_per_block, ET.cols_per_block, ET.data(),
                                C.data());

    // Update V matrix
    V = V.initialize_v(D);

    // Assuming D is stored on host
    free(D);
  }

#ifdef P_BENCHMARK
  // Stop the timer (before IO)
  auto end = std::chrono::high_resolution_clock::now();

  // Output runtime
  std::chrono::duration<double> elapsed = end - start;
  if (rank == 0)
    std::cout << "Runtime: " << elapsed.count() << " seconds" << std::endl;
#endif

  // Output cluster assignments
  std::string prefix = std::string(data_path);
  std::string suffix = "_out";
  V.save_assignments((prefix + suffix).c_str());
}

}  // namespace popcorn
