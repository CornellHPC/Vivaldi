// Local imports
#include "cluster.hh"
#include "kernel/polynomial_kernel.cuh"
#include "mat/dense_mat.hh"
#include "mat/sparse_mat.hh"
#include "utils/utils.hh"

namespace popcorn {

void cluster(char *data_path, int m, int n, int k, MPI_Comm comm) {
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
  // TODO: pull mb from the matrix instead of passing
  auto K = P.gemm(PT);
  auto poly_kernel = PolynomialKernel(1.0f, 1.0f, 2.0f);
  K.apply(poly_kernel);
  K.print("K");

  // Initialize the V matrix
  auto V = SparseMat::initialize_v(m, k, comm);
  V.print("V");

  // Perform SpMM(VK)
  auto O = V.spmm(K);
  O.print("O");
}

} // namespace popcorn
