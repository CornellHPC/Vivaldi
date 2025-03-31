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

typedef std::chrono::seconds s;
typedef std::chrono::milliseconds ms;
using hrc = std::chrono::high_resolution_clock;
// using ms = std::chrono::milliseconds;

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

  auto io_start = hrc::now();
  // Load the original data with SLATE, this will be transposed
  auto PT = DenseMat::load_from_file(data_path, m, n, comm);
  auto io_elapsed = get_time_elapsed(io_start);
#ifdef P_DEBUG
  PT.print("PT");
#endif

#ifdef P_BENCHMARK
  // Start the timer (after IO)
  auto start = std::chrono::high_resolution_clock::now();
  double k_elapsed = 0;
  double vi_elapsed = 0;
  double e_elapsed = 0;
  double vk_elapsed = 0;
  double c_elapsed = 0;
  double vr_elapsed = 0;
  double d_elapsed = 0;
#endif

  // Transpose back to obtain the original matrix
  auto P = PT.transpose();
#ifdef P_DEBUG
  P.print("P");
#endif

#ifdef P_BENCHMARK
  auto k_start = std::chrono::high_resolution_clock::now();
#endif

  // Compute the K matrix
  // TODO: make gamma, c, r as IO input
  auto K = P.gemm(PT);
  auto poly_kernel = PolynomialKernel(1.0f, 1.0f, 1.0f);
  K.apply(poly_kernel);

#ifdef P_BENCHMARK
  k_elapsed += std::chrono::duration_cast<ms>(
                   std::chrono::high_resolution_clock::now() - k_start)
                   .count();
#endif

#ifdef P_DEBUG
  K.print("K");
#endif

#ifdef P_BENCHMARK
  auto vi_start = std::chrono::high_resolution_clock::now();
#endif

  // Initialize the V matrix
  auto V = SparseMat::initialize_v(m, k, comm);

#ifdef P_BENCHMARK
  vi_elapsed = get_time_elapsed(vi_start);
#endif

#ifdef P_DEBUG
  V.print("V");
#endif

  for (int i = 0; i < 100; ++i) {
#ifdef P_BENCHMARK
    auto e_start = std::chrono::high_resolution_clock::now();
#endif

    // Perform SpMM(VK)
    auto ET = V.spmm(K);

#ifdef P_BENCHMARK
    e_elapsed += get_time_elapsed(e_start);
#endif

#ifdef P_DEBUG
    if (i == 0)
      ET.print("ET");
#endif

#ifdef P_BENCHMARK
    auto c_start = std::chrono::high_resolution_clock::now();
#endif

    // Compute the centroid norms
    auto C = DenseMat::initialize_cnorm(V, ET);

#ifdef P_BENCHMARK
    c_elapsed += std::chrono::duration_cast<ms>(
                     std::chrono::high_resolution_clock::now() - c_start)
                     .count();
#endif

#ifdef P_BENCHMARK
auto vr_start = std::chrono::high_resolution_clock::now();
#endif

#ifdef P_BENCHMARK
    auto d_start = std::chrono::high_resolution_clock::now();
#endif

    // Compute the D matrix
    auto dist_kernel = DistKernel();
    auto D = dist_kernel.kernel(ET.rows_per_block, ET.cols_per_block, ET.data(),
                                C.data());

#ifdef P_BENCHMARK
    d_elapsed += std::chrono::duration_cast<ms>(
                     std::chrono::high_resolution_clock::now() - d_start)
                     .count();
#endif

    // Update V matrix
    V = V.initialize_v(D);

#ifdef P_BENCHMARK
    vr_elapsed += std::chrono::duration_cast<ms>(
                      std::chrono::high_resolution_clock::now() - vr_start)
                      .count();
#endif

    // Assuming D is stored on host
    free(D);
  }

#ifdef P_BENCHMARK
  // Stop the timer (before IO)
  auto end = std::chrono::high_resolution_clock::now();

  // Output runtime
  double elapsed = std::chrono::duration_cast<ms>(end - start).count();
  if (rank == 0) {
    std::cout << "IO: " << io_elapsed << std::endl;
    std::cout << "K: " << k_elapsed << std::endl;
    std::cout << "VI: " << vi_elapsed << std::endl;
    std::cout << "E: " << e_elapsed << std::endl;
    std::cout << "C: " << c_elapsed << std::endl;
    std::cout << "VR total, including D: " << vr_elapsed << std::endl;
    std::cout << "D: " << d_elapsed << std::endl;
    std::cout << "Elapsed: " << elapsed << std::endl;
  }
#endif

  // Output cluster assignments
  std::string prefix = std::string(data_path);
  std::string suffix = "_out";
  V.save_assignments((prefix + suffix).c_str());
}

}  // namespace popcorn
