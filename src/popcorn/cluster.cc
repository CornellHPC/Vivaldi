// C++ standard imports
#include <cstdlib>

// Local imports
#include "cluster.hh"
#include "kernel/argmin_kernel.cuh"
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

  if (rank == 0)
#ifdef CUDA
    std::cout << "Running on: CUDA" << std::endl;
  wake_gpus(rank);
#else
    std::cout << "CUDA is unavailable. Some things may not work properly."
              << std::endl;
#endif

  // Cull points to make matrix evenly divisible
  int grid_dim = square_grid_dim(comm);
  m -= m % grid_dim;

  if (rank == 0)
    std::cout << "Reading data from " << data_path << std::endl;

  // Load the point matrix
  auto P = DenseMat::load_from_file(data_path, m, n, comm);
  P.print("P");

  // Make a transposed copy of point matrix
  auto PT = P.transpose();
  PT.print("PT");

  // Compute the K matrix
  // TODO: make gamma, c, r as IO input
  auto K = P.gemm(PT);
  auto poly_kernel = PolynomialKernel(1.0f, 1.0f, 1.0f);
  K.apply(poly_kernel);
  K.print("K");

  // Initialize the V matrix
  auto V = SparseMat::initialize_v(m, k, comm);
  V.print("V");

  for (int i = 0; i < 100; ++i) {
    // Perform SpMM(VK)
    auto ET = V.spmm(K);
    ET.print("ET");

    // Compute the centroid norms
    auto C = DenseMat::initialize_cnorm(V, ET);

    // Compute the D matrix
    auto dist_kernel = DistKernel();
    auto D = dist_kernel.kernel(ET.rows_per_block, ET.cols_per_block, ET.data(),
                                C.data());

    // Update V matrix
    V.initialize_v(m, k, ET.cols_per_block, ET.rows_per_block, D);

    // Assuming D is stored on host
    free(D);
  }

  // TODO: Output using V here
}

}  // namespace popcorn
